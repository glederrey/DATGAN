#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from datgan.synthesizer.losses.GANLoss import GANLoss


class WGANLoss(GANLoss):
    """
    Wasserstein loss function
    """

    def __init__(self, metadata, var_order, name="WGANLoss"):
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

    def gen_loss(self, synth_output, transformed_orig, transformed_synth, l2_reg):
        """
        Compute the Wasserstein distance and the kl divergence for the generator.

        Parameters
        ----------
        synth_output: tf.Tensor
            Output of the discriminator on the synthetic data
        transformed_orig: tf.Tensor
            Original data that have been encoded and transformed into a tensor
        transformed_synth: tf.Tensor
            Synthetic data that have been encoded and transformed into a tensor


        Returns
        -------
        tf.Tensor:
            Final loss function for the generator. (to be passed to the optimizer)
        """

        wass_loss = tf.negative(tf.reduce_mean(synth_output))

        kl = self.kl_div(transformed_orig, transformed_synth)

        loss = wass_loss + kl + l2_reg

        self.logs['generator']['gen_loss'] = wass_loss
        self.logs['generator']['kl_div'] = kl
        self.logs['generator']['reg_loss'] = l2_reg
        self.logs['generator']['loss'] = loss

        return loss

    def discr_loss(self, orig_output, synth_output, l2_reg):
        """
        Compute the Wasserstein distance for the discriminator.

        Parameters
        ----------
        orig_output: tf.Tensor
            Output of the discriminator on the original data
        synth_output: tf.Tensor
            Output of the discriminator on the synthetic data

        Returns
        -------
        tf.Tensor:
            Final loss function for the discriminator. (to be passed to the optimizer)
        """

        real_loss = tf.reduce_mean(orig_output)
        fake_loss = tf.reduce_mean(synth_output)
        loss = fake_loss - real_loss + l2_reg

        # Log stuff
        self.logs['discriminator']['loss_real'] = real_loss
        self.logs['discriminator']['loss_fake'] = fake_loss
        self.logs['discriminator']['reg_loss'] = l2_reg
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

        for l in ['loss_real', 'loss_fake', 'loss', 'reg_loss']:
            logs['discriminator'][l] = []

        logs['generator'] = {}

        for l in ['gen_loss', 'kl_div', 'loss', 'reg_loss']:
            logs['generator'][l] = []
