#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from datgan.synthesizer.losses.GANLoss import GANLoss


class SGANLoss(GANLoss):
    """
    Cross-entropy loss function
    """

    def __init__(self, metadata, var_order, name="SGANLoss"):
        """
        Initialize the class

        Parameters
        ----------
        metadata: dict
            Metadata for the columns (used when computing the kl divergence)
        var_order: list
            Ordered list of the variables (
        name: string
            Name of the loss function
        """
        super().__init__(metadata, var_order, name=name)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()

    def gen_loss(self, synth_output, transformed_orig, transformed_synth, l2_reg):
        """
        Compute the cross-entropy loss and the kl divergence for the generator. It also adds the L2 regularization to
        return the final loss

        Parameters
        ----------
        synth_output: tf.Tensor
            Output of the discriminator on the synthetic data
        transformed_orig: tf.Tensor
            Original data that have been encoded and transformed into a tensor
        transformed_synth: tf.Tensor
            Synthetic data that have been encoded and transformed into a tensor
        l2_reg: tf.Tensor
            Loss for the L2 regularization

        Returns
        -------
        tf.Tensor:
            Final loss function for the generator. (to be passed to the optimizer)
        """
        cr_entr = self.cross_entropy(tf.ones_like(synth_output), synth_output)
        kl = self.kl_div(transformed_orig, transformed_synth)

        loss = cr_entr + kl + l2_reg

        self.logs['generator']['gen_loss'] = cr_entr
        self.logs['generator']['kl_div'] = kl
        self.logs['generator']['reg_loss'] = l2_reg
        self.logs['generator']['loss'] = loss

        return loss

    def discr_loss(self, orig_output, synth_output, l2_reg):
        """
        Compute the cross-entropy loss for the discriminator. It also adds the L2 regularization to return the final
        loss.

        Parameters
        ----------
        orig_output: tf.Tensor
            Output of the discriminator on the original data
        synth_output: tf.Tensor
            Output of the discriminator on the synthetic data
        l2_reg: tf.Tensor
            Loss for the L2 regularization

        Returns
        -------
        tf.Tensor:
            Final loss function for the discriminator. (to be passed to the optimizer)
        """

        real_loss = self.cross_entropy(tf.ones_like(orig_output), orig_output)
        fake_loss = self.cross_entropy(tf.zeros_like(synth_output), synth_output)
        discr_loss = 0.5 * real_loss + 0.5 * fake_loss
        self.logs['discriminator']['discr_loss'] = discr_loss

        # Accuracy
        self.logs['discriminator']['acc_orig'] = tf.reduce_mean(tf.cast(orig_output > 0.5, tf.float32))
        self.logs['discriminator']['acc_synth'] = tf.reduce_mean(tf.cast(synth_output < 0.5, tf.float32))

        self.logs['discriminator']['reg_loss'] = l2_reg

        loss = discr_loss + l2_reg
        self.logs['discriminator']['loss'] = loss

        return loss

    def prepare_logs(self, logs):
        """
        Prepare the logs based on the computation of both loss functions. Used to help when saving logged values.

        Parameters
        ----------
        logs: dict
            Empty dictionary.
        """

        logs['discriminator'] = {}

        for l in ['discr_loss', 'acc_orig', 'acc_synth', 'reg_loss', 'loss']:
            logs['discriminator'][l] = []

        logs['generator'] = {}

        for l in ['gen_loss', 'kl_div', 'reg_loss', 'loss']:
            logs['generator'][l] = []
