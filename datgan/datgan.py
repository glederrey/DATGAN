#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import pickle
import tarfile
import numpy as np
import tensorflow as tf
from functools import partial
from datetime import datetime

#from datgan.utils.utils import ClipCallback
#from datgan.synthesizer.DATGANSynthesizer import DATGANSynthesizer
#from datgan.utils.trainer import GANTrainerClipping, SeparateGANTrainer
from datgan.utils.utils import elapsed_time
from datgan.utils.data import EncodedDataset
from datgan.synthesizer.synthesizer import Synthesizer
from datgan.utils.dag import verify_dag, get_order_variables, linear_DAG


class DATGAN:
    """
    Main class for DATGAN synthesizer.

    Methods
    -------
        preprocessing:
            Preprocess the original data

        fit:
            Fit the DATGAN model to the encoded data

        sample:
            Sample the synthetic data from the trained DATGAN model

        save:
            Save the model to load it later.

        load:
            Load the model.

    """

    def __init__(self, loss_function=None, label_smoothing='TS', output='output', gpu=True, num_epochs=100,
                 batch_size=500, save_checkpoints=True, restore_session=True, learning_rate=None, z_dim=200,
                 num_gen_rnn=100, num_gen_hidden=50, num_dis_layers=1, num_dis_hidden=100, noise=0.2, l2norm=0.00001,
                 verbose=0):
        """
        Constructs all the necessary attributes for the DATGAN class.

        Parameters
        ----------
            loss_function: str, default None
                Name of the loss function to be used. If not specified, it will choose between 'WGAN' and 'WGGP'
                depending on the ratio of continuous and categorical columns. Only accepts the values 'SGAN', 'WGAN',
                and 'WGGP'.
            label_smoothing: str, default 'TS'
                Type of label smoothing. Only accepts the values 'TS', 'OS', and 'NO'.
            output: str, default 'output'
                Path to store the model and its artifacts.
            gpu: bool, default True
                Use the first available GPU if there's one and tensorflow has been built with cuda.
            num_epochs: int, default 100
                Number of epochs to use during training.
            batch_size: int, default 500
                Size of the batch to feed the model at each step.
            save_checkpoints: bool, default True
                Whether to store checkpoints of the model after each training epoch.
            restore_session: bool, default True
                Whether continue training from the last checkpoint.
            learning_rate: float, default None
                Learning rate. If set to None, the value will be set according to the chosen loss function.
            z_dim: int, default 200
                Dimension of the noise vector used as an input to the generator.
            num_gen_rnn: int, default 100
                Size of the hidden units in the LSTM cell.
            num_gen_hidden: int, default 50
                Size of the hidden layer used on the output of the generator to act as a convolution.
            num_dis_layers: int, default 1
                Number of layers for the discriminator.
            num_dis_hidden: int, default 100
                Size of the hidden layers in the discriminator.
            noise: float, default 0.2
                Upper bound to the gaussian noise added to with the label smoothing. (only used if label_smoothing is
                set to 'TS' or 'OS')
            l2norm: float, default 0.00001
                L2 reguralization coefficient when computing the standard GAN loss.
            verbose: int, default 0
                Level of verbose. 0 means nothing, 1 means that some details will be printed, 2 is mostly used for
                debugging purpose.

            Raises
            ------
                ValueError
                    If the parameter loss_function is not correctly defined.
                ValueError
                    If the parameter label_smoothing is not correctly defined.
        """

        self.output = output

        # Some paths and directories
        self.log_dir = os.path.join(self.output, 'logs')
        self.model_dir = os.path.join(self.output, 'model')
        self.restore_path = os.path.join(self.model_dir, 'checkpoint')

        self.data_dir = None

        # Training parameters
        self.num_epochs = num_epochs
        self.save_checkpoints = save_checkpoints
        self.restore_session = restore_session

        # Model parameters
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.l2norm = l2norm
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_hidden = num_gen_hidden
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden
        self.verbose = verbose

        # Specific parameters for the DATGAN
        self.label_smoothing = label_smoothing
        if self.label_smoothing not in ['TS', 'OS', 'NO']:
            raise ValueError("The variable 'label_smoothing' must take one of the following values: 'TS' (two-sided), "
                             "'OS' (one-sided), or 'NO' (none).")

        self.loss_function = loss_function
        if self.loss_function and self.loss_function not in ['SGAN', 'WGAN', 'WGGP']:
            raise ValueError("The variable 'loss_function' must take one of the following values: 'SGAN' (Standard GAN "
                             "loss function), 'WGAN' (Wasserstein loss function), or 'WGGP' (Wasserstein loss function"
                             "with gradient penalty).")

        # Check if there's a GPU available and tensorflow has been compiled with cuda
        if gpu:
            if len(tf.config.list_physical_devices('GPU')) > 0 and tf.test.is_built_with_cuda():
                os.environ['CUDA_VISIBLE_DEVICES'] = "0"

        # General variables that are defined in other functions
        self.metadata = None
        self.synthesizer = None
        self.encoded_data = None
        self.continuous_columns = None
        self.simple_dataset_predictor = None

        # Variables for the DAG (defined in the fit function)
        self.dag = None
        self.var_order = None
        self.n_sources = None

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                           Preprocessing the data                                           """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def preprocess(self, data, continuous_columns, preprocessed_data_path=None):
        """
        Preprocess the original data to transform it into a usable dataset.

        Parameters
        ----------
            data: pandas.DataFrame
                Original dataset
            continuous_columns: list[str]
                List of the names of the continuous columns
            preprocessed_data_path: str, default None
                Path to an existing preprocessor. If None is given, the model will preprocess the data and save it under
                self.output + '/encoded_data'.

        Raises
        ------
            FileNotFoundError
                If the files 'preprocessed_data.pkl' and 'preprocessor.pkl' are not in the folder given in the variable
                preprocessed_data_path

        Returns
        -------
        """
        # If the preprocessed_data_path is not given, we create it
        if not preprocessed_data_path:
            preprocessed_data_path = os.path.join(self.output, 'encoded_data')
            if self.verbose > 1:
                print("No path given. Saving the encoded data here: {}".format(preprocessed_data_path))

        if not os.path.exists(preprocessed_data_path):
            os.makedirs(preprocessed_data_path)

        self.data_dir = preprocessed_data_path

        # Load the existing preprocessor
        if os.path.exists(os.path.join(self.data_dir, 'encoded_data.pkl')):

            # Load the preprocessor and the preprocessed data
            with open(os.path.join(self.data_dir, 'encoded_data.pkl'), 'rb') as f:
                self.encoded_data = pickle.load(f)

            if self.verbose > 0:
                print("Preprocessed data have been loaded!")
        else:
            if self.verbose > 0:
                print("Preprocessing the data!")

            # Preprocess the original data
            self.encoded_data = EncodedDataset(data, continuous_columns, self.verbose)
            self.encoded_data.fit_transform(fitting=True)

            # Save them both
            with open(os.path.join(self.data_dir, 'encoded_data.pkl'), 'wb') as f:
                pickle.dump(self.encoded_data, f)

            # Verification for continuous mixture
            self.encoded_data.plot_continuous_mixtures(data, self.data_dir)

            if self.verbose > 0:
                print("Preprocessed data have been saved in '{}'".format(self.data_dir))

        # If the preprocessed_data_path is not given, we create it
        if not preprocessed_data_path:
            preprocessed_data_path = os.path.join(self.output, 'encoded_data')
            print("No path given. Saving the encoded data here: {}".format(preprocessed_data_path))

        if not os.path.exists(preprocessed_data_path):
            os.makedirs(preprocessed_data_path)

        self.data_dir = preprocessed_data_path

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                             Fitting the model                                              """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def fit(self, data, continuous_columns, dag=None, preprocessed_data_path=None):
        """
        Fit the DATGAN model to the original data and save it once it's finished training.
        Parameters
        ----------
        data: pandas.DataFrame
            Original dataset
        continuous_columns: list[str]
            List of the names of the continuous columns
        dag: networkx.DiGraph, default None
            Directed Acyclic Graph representing the relations between the variables. If no dag is provided, the
            algorithm will create a linear DAG.
        preprocessed_data_path: str, default None
            Path to an existing preprocessor. If None is given, the model will preprocess the data and save it under
            self.output + '/encoded_data'.
        """

        # Preprocess the original data
        self.preprocess(data, continuous_columns, preprocessed_data_path)
        self.metadata = self.encoded_data.metadata

        # Verify the integrity of the DAG and get the ordered list of variables for the Generator
        if not dag:
            self.dag = linear_DAG(data)
        else:
            self.dag = dag

        verify_dag(data, dag)
        self.var_order, self.n_sources = get_order_variables(dag)

        # Define the loss function if not defined yet
        if not self.loss_function:
            # more categorical columns than continuous
            if float(len(continuous_columns))/float(len(data.columns)) < 0.5:
                self.loss_function = 'WGAN'
            else:
                self.loss_function = 'WGGP'

        if not self.learning_rate:
            if self.loss_function == 'SGAN':
                self.learning_rate = 1e-3
            elif self.loss_function == 'WGAN':
                self.learning_rate = 2e-4
            elif self.loss_function == 'WGGP':
                self.learning_rate = 1e-4

        # Create the folders used to save the checkpoints of the model
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        start = datetime.now()
        start_t = time.perf_counter()

        # If we want to restart the training session, we need to find the pickle file
        if self.restore_session and os.path.exists(os.path.join(self.output, 'synthesizer.pkl')):

            # Load the pickle file
            with open(os.path.join(self.output, 'synthesizer.pkl'), 'rb') as infile:
                self.synthesizer = pickle.load(infile)

            # Update some stuff in the synthesizer
            self.synthesizer.save_checkpoints = self.save_checkpoints
            self.synthesizer.verbose = self.verbose

            # Check if we still need to train the model
            if self.num_epochs <= self.synthesizer.trained_epoch:
                if self.verbose > 0:
                    print("DATGAN model has already been trained for {} epochs.".format(self.synthesizer.trained_epoch))
                return
            else:
                if self.verbose > 0:
                    dt_string = start.strftime("%d/%m/%Y %H:%M:%S")
                    print("Continue the training of DATGAN at epoch {} ({}).".format(self.synthesizer.trained_epoch,
                                                                                     dt_string))
        else:
            # Load a new synthesizer
            if self.verbose > 0:
                dt_string = start.strftime("%d/%m/%Y %H:%M:%S")
                print("Start training DATGAN with the {} loss ({}).".format(self.loss_function, dt_string))

            self.synthesizer = Synthesizer(self.output, self.metadata, self.dag, self.batch_size, self.z_dim,
                                           self.noise, self.learning_rate, self.num_gen_rnn, self.num_gen_hidden,
                                           self.num_dis_layers, self.num_dis_hidden, self.label_smoothing,
                                           self.loss_function, self.var_order, self.n_sources, self.save_checkpoints,
                                           self.verbose)

        # Fit the model
        self.synthesizer.fit(self.encoded_data, self.num_epochs)

        end = datetime.now()
        if self.verbose > 0:
            dt_string = end.strftime("%d/%m/%Y %H:%M:%S")

            delta = time.perf_counter()-start_t
            str_delta = elapsed_time(delta)

            print("DATGAN has finished training ({}) - Training time: {}".format(dt_string, str_delta))

    def fit_old(self, data, dag, continuous_columns, preprocessed_data_path=None):
        """
        Fit the DATGAN model to the original data and save it once it's finished training.

        Parameters
        ----------
            data: pandas.DataFrame
                Original dataset
            dag: networkx.DiGraph
                Directed Acyclic Graph representing the relations between the variables
            continuous_columns: list[str]
                List of the names of the continuous columns
            preprocessed_data_path: str, default None
                Path to an existing preprocessor. If None is given, the model will preprocess the data and save it under
                self.output + '/encoded_data'.
        """

        # Preprocess the original data
        self.preprocess(data, continuous_columns, preprocessed_data_path)

        # Verify the integrity of the DAG and get the ordered list of variables for the Generator
        self.dag = dag
        verify_dag(data, dag)
        self.var_order, self.n_sources = get_order_variables(dag)

        # Define the loss function if not defined yet
        if not self.loss_function:
            # more categorical columns than continuous
            if float(len(continuous_columns))/float(len(data.columns)) < 0.5:
                self.loss_function = 'WGAN'
            else:
                self.loss_function = 'WGGP'

        # Define the trainer based on the loss function
        if self.loss_function == 'SGAN':
            self.trainer = GANTrainerClipping
            if not self.learning_rate:
                self.learning_rate = 1e-3
        elif self.loss_function == 'WGAN':
            self.trainer = partial(SeparateGANTrainer, g_period=3)
            if not self.learning_rate:
                self.learning_rate = 2e-4
        elif self.loss_function == 'WGGP':
            self.trainer = partial(SeparateGANTrainer, g_period=6)
            if not self.learning_rate:
                self.learning_rate = 1e-4

        self.metadata = self.preprocessor.metadata
        dataflow = DATGANDataFlow(self.transformed_data, self.metadata, self.var_order)
        batch_data = BatchData(dataflow, self.batch_size)
        input_queue = QueueInput(batch_data)

        self.model = self.get_model()

        trainer = self.trainer(
            input=input_queue,
            model=self.model
        )

        # Checking if previous training already exists
        session_init = None
        starting_epoch = 1
        if os.path.isfile(self.restore_path) and self.restore_session:
            print("Found an already existing model. Loading it!")

            session_init = SaverRestore(self.restore_path)
            with open(os.path.join(self.log_dir, 'stats.json')) as f:
                starting_epoch = json.load(f)[-1]['epoch_num'] + 1

        action = 'k' if self.restore_session else None
        logger.set_logger_dir(self.log_dir, action=action)

        callbacks = self.get_callbacks()

        steps_per_epoch = max(len(data) // self.batch_size, 1)

        # Actually train the model!
        trainer.train_with_defaults(
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            max_epoch=self.max_epoch,
            session_init=session_init,
            starting_epoch=starting_epoch
        )

        # Prepare the model for sampling the synthetic data
        self.prepare_sampling()

    def get_model(self):
        """
        Returns an instance of the model to train

        Returns
        -------
            DATGANSynthesizer:
                Model implemented in tensorflow

        """
        return DATGANSynthesizer(
            metadata=self.metadata,
            dag=self.dag,
            batch_size=self.batch_size,
            z_dim=self.z_dim,
            noise=self.noise,
            l2norm=self.l2norm,
            learning_rate=self.learning_rate,
            num_gen_rnn=self.num_gen_rnn,
            num_gen_hidden=self.num_gen_hidden,
            num_dis_layers=self.num_dis_layers,
            num_dis_hidden=self.num_dis_hidden,
            label_smoothing=self.label_smoothing,
            loss_function=self.loss_function,
            var_order=self.var_order
        )

    def get_callbacks(self):
        """
        Returns the callbacks for the training. The list depends on the loss function.

        Returns
        -------
            list:
                list of callbacks

        """

        callbacks = []
        # Callback to save the model
        if self.save_checkpoints:
            callbacks.append(ModelSaver(checkpoint_dir=self.model_dir))

        # Callback to clip the gradient's values when training the model with the WGAN loss
        if self.loss_function == 'WGAN':
            callbacks.append(ClipCallback())

        return callbacks

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                                  Sampling                                                  """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def sample(self, num_samples, sampling='SS'):
        """

        Parameters
        ----------
            num_samples: int
                Number of sample to provide
            sampling: str
                Type of sampling to use. Only accepts the following values: 'SS', 'SA', 'AS', and 'AA'

        Returns
        -------
            pandas.DataFrame:
                Synthetic dataset of 'num_samples' rows

        """
        if not self.preprocessor:
            print("Loading Preprocessor!")
            # Load preprocessor
            with open(os.path.join(self.data_dir, 'preprocessor.pkl'), 'rb') as f:
                self.preprocessor = pickle.load(f)

        if sampling not in ['SS', 'AS', 'SA', 'AA']:
            raise ValueError("'sampling' must take value 'SS', 'AS', 'SA', or 'AA'!")

        self.preprocessor.set_sampling_technique(sampling)
        self.metadata = self.preprocessor.metadata

        max_iters = (num_samples // self.batch_size)

        results = []
        for idx, o in enumerate(self.simple_dataset_predictor.get_result()):
            results.append(o[0])
            if idx == max_iters:
                break

        results = np.concatenate(results, axis=0)
        # Reduce results to num_samples
        results = results[:num_samples]

        ptr = 0
        features = {}
        # Go through all variables
        for col_id, col in enumerate(self.var_order):
            # Get info
            col_info = self.metadata['details'][col]
            if col_info['type'] == 'category':
                features[col] = results[:, ptr:ptr + col_info['n']]
                ptr += col_info['n']

            elif col_info['type'] == 'continuous':

                n_modes = col_info['n']

                val = results[:, ptr:ptr + n_modes]
                ptr += n_modes

                pro = results[:, ptr:ptr + n_modes]
                ptr += n_modes

                features[col] = np.concatenate([val, pro], axis=1)

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`continuous`. Instead it was {}.".format(col_id, col_info['type'])
                )

        return self.preprocessor.reverse_transform(features)[:num_samples].copy()

    def prepare_sampling(self):
        """
        Prepare model to generate samples.
        """

        if self.model is None:
            self.model = self.get_model()

        predict_config = PredictConfig(
            session_init=SaverRestore(self.restore_path),
            model=self.model,
            input_names=['z'],
            output_names=['output', 'z'],
        )

        self.simple_dataset_predictor = SimpleDatasetPredictor(
            predict_config,
            RandomZData((self.n_sources, self.batch_size, self.z_dim))
        )

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                              Other functions                                               """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    @classmethod
    def load(cls, path, name):
        """
        Load a pretrained model from a given path.

        Parameters
        ----------
            path: str
                Path where the pretrained model is located
            name: str
                Name of the pretrained model

        Returns
        -------
            datgan.DATGAN:
                Pretrained model
        """
        with tarfile.open(path + name + '.tar.gz', 'r:gz') as tar_handle:
            destination_dir = os.path.dirname(tar_handle.getmembers()[0].name)
            tar_handle.extractall()

        with open('{}/{}.pickle'.format(destination_dir, name), 'rb') as f:
            instance = pickle.load(f)

        instance.prepare_sampling()
        return instance

    def save(self, name, force=False):
        """
        Save the fitted model in the given path.

        Parameters
        ----------
        name: str
            Name of the current model (path has to be included)
        force: bool
            Boolean used to overwrite the existing model
        """
        if os.path.exists(self.output) and not force:
            print('The indicated path already exists. Use `force=True` to overwrite.')
            return

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        model = self.model
        dataset_predictor = self.simple_dataset_predictor

        self.model = None
        self.simple_dataset_predictor = None

        with open('{}/{}.pickle'.format(self.output, name), 'wb') as f:
            pickle.dump(self, f)

        self.model = model
        self.simple_dataset_predictor = dataset_predictor

        self.tar_folder(self.output + name + '.tar.gz')

        print('Model saved successfully.')

    def tar_folder(self, tar_name):
        """
        Generate a tar of the output

        Parameters
        ----------
            tar_name: str
                Name of the tar
        """
        with tarfile.open(tar_name, 'w:gz') as tar_handle:
            for root, dirs, files in os.walk(self.output):
                for file_ in files:
                    tar_handle.add(os.path.join(root, file_))

            tar_handle.close()