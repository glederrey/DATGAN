#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File that describes the Synthesizer for the DATGAN
"""
import os
import json
import time
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from datgan.utils.utils import elapsed_time
from datgan.synthesizer.generator import Generator
from datgan.synthesizer.discriminator import Discriminator


class Synthesizer:
    """
    Synthesizer for the DATGAN model
    """

    def __init__(self, output, metadata, dag, batch_size, z_dim, noise, learning_rate, num_gen_rnn, num_gen_hidden,
                 num_dis_layers, num_dis_hidden, label_smoothing, loss_function, var_order, n_sources, save_checkpoints,
                 restore_session, verbose):
        """
        Constructs all the necessary attributes for the DATGANSynthesizer class.
        Parameters
        ----------
        output: str
            Output path
        metadata: dict
            Information about the data.
        dag: networkx.DiGraph
            Directed Acyclic Graph provided by the user.
        batch_size: int
            Size of the batch to feed the model at each step. Defined in the DATGAN class.
        z_dim: int
            Dimension of the noise vector used as an input to the generator. Defined in the DATGAN class.
        noise: float
            Upper bound to the gaussian noise added to with the label smoothing. (only used if label_smoothing is
            set to 'TS' or 'OS') Defined in the DATGAN class.
        learning_rate: float
            Learning rate. Defined in the DATGAN class.
        num_gen_rnn: int
            Size of the hidden units in the LSTM cell. Defined in the DATGAN class.
        num_gen_hidden: int
            Size of the hidden layer used on the output of the generator to act as a convolution. Defined in the DATGAN
            class.
        num_dis_layers: int
            Number of layers for the discriminator. Defined in the DATGAN class.
        num_dis_hidden: int
            Size of the hidden layers in the discriminator. Defined in the DATGAN class.
        label_smoothing: str
            Type of label smoothing. Defined in the DATGAN class.
        loss_function: str
            Name of the loss function to be used. Defined in the DATGAN class.
        var_order: list[str]
            Ordered list for the variables. Used in the Generator.
        n_sources: int
            Number of source nodes in the DAG.
        save_checkpoints: bool, default True
            Whether to store checkpoints of the model after each training epoch.
        restore_session: bool, default True
            Whether continue training from the last checkpoint.
        verbose: int
            Level of verbose.
        """

        self.output = output
        self.metadata = metadata
        self.dag = dag
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_hidden = num_gen_hidden
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden
        self.label_smoothing = label_smoothing
        self.loss_function = loss_function
        self.var_order = var_order
        self.n_sources = n_sources
        self.save_checkpoints = save_checkpoints
        self.restore_session = restore_session
        self.verbose = verbose

        # Parameter used for the WGGP loss function
        self.lambda_ = 10

        # Checkpoints
        self.checkpoint = None
        self.checkpoint_manager = None
        self.checkpoint_dir = self.output + 'checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        # Other parameters
        self.generator = None
        self.discriminator = None
        self.optimizerD = None
        self.optimizerG = None
        self.data = None
        self.logging = {'generator': {}, 'discriminator': {}}

        if self.loss_function == 'SGAN':
            self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        else:
            raise NotImplementedError("OTHER LOSS FUNCTIONS ARE NOT IMPLEMENTED YET!")

        self.kl = tf.keras.losses.KLDivergence()

        # Training
        self.g_period = 1
        if self.loss_function in ['WGAN', 'WGGP']:
            self.g_period = 5

    def fit(self, encoded_data, num_epochs):
        """
        Fit the synthesizer
        Parameters
        ----------
        encoded_data: dict
            Dictionary of the encoded data to be transformed into a tensor.
        num_epochs: int
            Number of epochs
        """

        self.initialize()

        self.data = (tf.data.Dataset.from_tensor_slices(encoded_data)
                     .shuffle(self.metadata['len'], reshuffle_each_iteration=True)
                     .batch(self.batch_size, drop_remainder=True))

        # Verbose stuff
        if self.verbose == 1:
            iterable_epochs = tqdm(range(self.checkpoint.epoch.numpy(), num_epochs), desc="Training DATGAN")
        else:
            iterable_epochs = range(self.checkpoint.epoch.numpy(), num_epochs)

        if self.verbose == 2:
            # Time for training each epoch
            time_epoch = []

        # Keep track of all the iterations for the WGAN and WGGP loss
        iter_ = 0

        for epoch in iterable_epochs:

            if self.verbose == 2:
                iterable_steps = tqdm(self.data, desc="Epoch {}/{}".format(epoch + 1, num_epochs))
                start_time = time.perf_counter()
            else:
                iterable_steps = self.data

            # We want to get the average value per epoch. => we collect the logs of the discriminator and the
            # generator in these dictionaries.
            discr_logging = {}
            gen_logging = {}

            for batch in iterable_steps:

                d_logs_dct, g_logs_dct = self.do_step(batch, iter_)

                # Save values for the discriminator
                for key in d_logs_dct:
                    if key not in discr_logging:
                        discr_logging[key] = [d_logs_dct[key].numpy()]
                    else:
                        discr_logging[key].append(d_logs_dct[key].numpy())

                # Save values for the generator
                for key in g_logs_dct:
                    if key not in gen_logging:
                        gen_logging[key] = [g_logs_dct[key].numpy()]
                    else:
                        gen_logging[key].append(g_logs_dct[key].numpy())

                # Update the iterations
                iter_ += 1

            # Log some values
            for key in discr_logging:
                if epoch == 0:
                    self.logging['discriminator'][key] = [np.mean(discr_logging[key], dtype=np.float64)]
                else:
                    self.logging['discriminator'][key].append(np.mean(discr_logging[key], dtype=np.float64))

            for key in gen_logging:
                if epoch == 0:
                    self.logging['generator'][key] = [np.mean(gen_logging[key], dtype=np.float64)]
                else:
                    self.logging['generator'][key].append(np.mean(gen_logging[key], dtype=np.float64))

            # Save the log file
            with open(os.path.join(self.output, 'logging.json'), 'w') as outfile:
                json.dump(self.logging, outfile)

            self.checkpoint.epoch.assign_add(1)

            # Save a checkpoint or at least the last model
            if self.save_checkpoints or int(self.checkpoint.epoch.numpy()) == num_epochs:
                self.checkpoint_manager.save()

            if self.verbose == 2:
                end_time = time.perf_counter()
                print("\033[1mEpoch {}\033[0m finished. time: {:.2f} seconds".format(int(self.checkpoint.epoch.numpy()),
                                                                                     end_time - start_time))
                print("Generator:")
                print("  {} loss: {:.2e}".format(self.loss_function, self.logging['generator']['gen_loss'][-1]))
                print("  KL div.: {:.2e}".format(self.logging['generator']['kl'][-1]))
                if self.loss_function == 'SGAN':
                    print("  Reg. loss: {:.2e}".format(self.logging['generator']['reg_loss'][-1]))
                print("\033[1m  Gen. loss: {:.2e}\033[0m".format(self.logging['generator']['loss'][-1]))

                print("Discriminator:")
                if self.loss_function == 'SGAN':
                    print("  Accuracy original: {:.1f}%".format(self.logging['discriminator']['acc_orig'][-1] * 100))
                    print("  Accuracy synthetic: {:.1f}%".format(self.logging['discriminator']['acc_synth'][-1] * 100))
                    print("  SGAN loss: {:.2e}".format(self.logging['discriminator']['d_loss'][-1]))
                    print("  Reg. loss: {:.2e}".format(self.logging['discriminator']['reg_loss'][-1]))

                else:
                    print("  Wasserstein loss on real data: {:.2e}".format(
                        self.logging['discriminator']['loss_real'][-1]))
                    print("  Wasserstein loss on fake data: {:.2e}".format(
                        self.logging['discriminator']['loss_fake'][-1]))

                    if self.loss_function == 'WGGP':
                        print("  Wasserstein loss: {:.2e}".format(self.logging['discriminator']['wass_loss'][-1]))
                        print("  Grad. pen.: {:.2e}".format(self.logging['discriminator']['grad_pen'][-1]))

                print("\033[1m  Discr. loss: {:.2e}\033[0m".format(self.logging['discriminator']['loss'][-1]))

                # Remaining training time
                if self.checkpoint.epoch.numpy() < num_epochs:
                    time_epoch.append(end_time - start_time)
                    if len(time_epoch) < 5:
                        avg = np.mean(time_epoch)
                    else:
                        avg = np.mean(time_epoch[-5:])
                    remaining_time = avg * (num_epochs - self.checkpoint.epoch.numpy())
                    print("Remaining training time around \033[1m{}\033[0m.".format(elapsed_time(remaining_time)))

                print()

    def initialize(self):
        """
        Initialize different stuff for fitting the DATGAN model
        """

        # Load the generator and the discriminator
        self.generator = Generator(self.metadata, self.dag, self.batch_size, self.z_dim, self.num_gen_rnn,
                                   self.num_gen_hidden, self.var_order, self.loss_function, self.verbose)

        self.discriminator = Discriminator(self.num_dis_layers, self.num_dis_hidden, self.loss_function)

        # Get the optimizer depending on the loss function
        if self.loss_function == 'SGAN':
            self.optimizerG = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.9)
            # Smaller learning rate for the discriminator
            self.optimizerD = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.9)
        elif self.loss_function == 'WGAN':
            pass
        elif self.loss_function == 'WGGP':
            pass

        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0),
                                              generator_optimizer=self.optimizerG,
                                              discriminator_optimizer=self.optimizerD,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        # Restore from checkpoints
        self.restore_models()

    def restore_models(self):
        if self.restore_session:
            # Restore the models
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            if self.checkpoint_manager.latest_checkpoint and self.verbose > 0:
                print("Restored models from epoch {:d}.".format(int(self.checkpoint.epoch.numpy())))

                # Reload the logs
                with open(os.path.join(self.output, 'logging.json'), 'r') as infile:
                    self.logging = json.load(infile)

    def do_step(self, batch, current_iter):
        """
        Do one step of the optimization process

        Parameters
        ----------
        batch: dict
            Dictionary of tensors containing a batch of the original data
        current_iter: int
            Current iteration
        """

        noise = tf.random.normal([self.n_sources, self.batch_size, self.z_dim])

        train_gen = (current_iter % self.g_period == 0)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Only train the generator every g_period steps
            synth = self.generator(noise, training=train_gen)

            # Transform the data
            transformed_orig, transformed_synth = self.transform_data(batch, synth)

            # Use the discriminator on the original and synthetic data
            orig_output = self.discriminator(transformed_orig, training=True)
            synth_output = self.discriminator(transformed_synth, training=True)

            # Compute loss functions
            if train_gen:
                gen_logs = self.generator_loss(synth_output, transformed_orig, transformed_synth)

                # For the SGAN model, we use regularization
                if self.loss_function == 'SGAN':
                    reg_loss = tf.reduce_sum(self.generator.losses)
                    gen_logs['reg_loss'] = reg_loss
                    gen_logs['loss'] = gen_logs['loss'] + reg_loss
            else:
                gen_logs = {}

            discr_logs = self.discriminator_loss(orig_output, synth_output)

            # For the SGAN model, we use regularization
            if self.loss_function == 'SGAN':
                reg_loss = tf.reduce_sum(self.discriminator.losses)
                discr_logs['d_loss'] = discr_logs['loss']
                discr_logs['reg_loss'] = reg_loss
                discr_logs['loss'] = discr_logs['d_loss'] + reg_loss

        # Apply one step of optimization for the Generator
        if train_gen:
            gen_grad = gen_tape.gradient(gen_logs['loss'], self.generator.trainable_variables)
            self.optimizerG.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

        # Apply one step of optimization for the Discriminator
        discr_grad = disc_tape.gradient(discr_logs['loss'], self.discriminator.trainable_variables)
        self.optimizerD.apply_gradients(zip(discr_grad, self.discriminator.trainable_variables))

        return discr_logs, gen_logs

    def generator_loss(self, synth_output, transformed_orig, transformed_synth):
        """
        Return the loss of the generator
        Parameters
        ----------
        synth_output: tf.Tensor
            Output of the discriminator on the synthetic data
        transformed_orig: tf.Tensor
            Tensor of the encoded original data (used for the KL div)
        transformed_synth: tf.Tensor
            Tensor of the encoded synthetic data (used for the KL div)

        Returns
        -------
        dict
            Logs of the loss function for the generator
        """

        logs = {}

        if self.loss_function == 'SGAN':
            loss = self.cross_entropy(tf.ones_like(synth_output), synth_output)
            logs['gen_loss'] = loss

        # KL divergence
        kl = self.kl_div(transformed_orig, transformed_synth)

        # Log some stuff
        logs['kl'] = kl
        logs['loss'] = loss + kl

        return logs

    def kl_div(self, original, synthetic):
        """
        Compute the KL divergence on the right variables.

        Parameters
        ----------
        original: tf.Tensor
            Tensor for the original encoded variables
        synthetic: tf.Tensor
            Tensor for the synthetic encoded variables

        Returns
        -------
        tf.Tensor:
            Sum of the KL divergences for the right variables

        """
        # KL loss
        kl_div = 0.0
        ptr = 0

        # Go through all variables
        for col_id, col in enumerate(self.var_order):
            # Get info
            col_info = self.metadata['details'][col]

            if col_info['type'] == 'continuous':
                # Skip the value. We only compute the KL on the probability vector
                ptr += col_info['n']

            dist = tf.reduce_sum(synthetic[:, ptr:ptr+col_info['n']], axis=0)
            dist = dist / tf.reduce_sum(dist)

            real = tf.reduce_sum(original[:, ptr:ptr+col_info['n']], axis=0)
            real = real / tf.reduce_sum(real)

            kl_div += self.kl(real, dist)
            ptr += col_info['n']

        return kl_div

    def discriminator_loss(self, orig_output, synth_output):
        """
        Return the loss of the discriminator

        Parameters
        ----------
        orig_output: tf.Tensor
            Output of the discriminator on the original data
        synth_output: tf.Tensor
            Output of the discriminator on the synthetic data

        Returns
        -------
        dict
            Logs of the loss function for the discriminator
        """

        logs = {}

        if self.loss_function == 'SGAN':
            real_loss = self.cross_entropy(tf.ones_like(orig_output), orig_output)
            fake_loss = self.cross_entropy(tf.zeros_like(synth_output), synth_output)
            discr_loss = 0.5*real_loss + 0.5*fake_loss

            # Log some stuff
            logs['loss'] = discr_loss
            logs['acc_orig'] = tf.reduce_mean(tf.cast(orig_output > 0.5, tf.float32))
            logs['acc_synth'] = tf.reduce_mean(tf.cast(synth_output < 0.5, tf.float32))

        return logs

    def transform_data(self, original, synthetic):
        """
        Transform the original data and the synthetic data in torch.Tensor.
        Parameters
        ----------
        original: dict
            Dictionary of the original encoded data
        synthetic: dict
            Dictionary of the synthetic encoded data
        Returns
        -------
        real: tf.Tensor
            Transformed original encoded data
        fake: tf.Tensor
            Transformed synthetic encoded data
        """

        real = []
        fake = []

        for col in self.var_order:
            # Get info
            col_info = self.metadata['details'][col]

            if col_info['type'] == 'category':

                # Synthetic data
                val_synth = synthetic[col]

                # Label smoothing
                if self.label_smoothing == 'TS':
                    noise = tf.random.uniform(val_synth.shape, minval=0, maxval=self.noise)
                    val_synth = (val_synth + noise) / tf.reduce_sum(val_synth + noise, keepdims=True, axis=1)

                fake.append(val_synth)

                # Original data
                val_orig = original[col]

                # Label smoothing
                if self.label_smoothing in ['TS', 'OS']:
                    noise = tf.random.uniform(val_orig.shape, minval=0, maxval=self.noise)
                    val_orig = (val_orig + noise) / tf.reduce_sum(val_orig + noise, keepdims=True, axis=1)

                real.append(val_orig)

            elif col_info['type'] == 'continuous':
                # Add values
                real.append(original[col][:, :col_info['n']])
                fake.append(synthetic[col][:, :col_info['n']])

                # Add probabilities
                real.append(original[col][:, col_info['n']:])
                fake.append(synthetic[col][:, col_info['n']:])

        return tf.concat(real, axis=1), tf.concat(fake, axis=1)

    def sample(self, n_samples):
        """
        Use the generator to sample data
        Parameters
        ----------
        n_samples: int
            Number of samples
        Returns
        -------
        samples: numpy.ndarray
            Matrix of encoded synthetic variables
        """

        steps = n_samples // self.batch_size + 1

        samples = {}
        for i in range(steps):
            # Compute the noise
            z = tf.random.normal([self.n_sources, self.batch_size, self.z_dim])

            # Generate data
            synth = self.generator(z)

            if i == 0:
                samples = synth
            else:
                for col in self.var_order:
                    samples[col] = tf.concat([samples[col], synth[col]], axis=0)

        for col in self.var_order:
            samples[col] = samples[col].numpy()[:n_samples, :]

        return samples
