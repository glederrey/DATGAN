#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from datgan.synthesizer.losses.GANLoss import GANLoss


class WGGPLoss(GANLoss):
    """
    Wasserstein loss with Gradient Penalty function
    """

    def __init__(self, metadata, var_order, name="WGGPLoss"):
        """
        Initialize the class

        Parameters
        ----------
        metadata: dict
            Metadata for the columns (used when computing the kl divergence)
        var_order: list
            Ordered list of the variables
        name: string
            Name of the loss function
        """
        super().__init__(metadata, var_order, name=name)

        self.lambda_ = tf.constant(10.0, dtype=tf.float32)

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
        l2_reg: tf.Tensor
            Loss for the L2 regularization

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

    def discr_loss(self, orig_output, synth_output, interp_grad, l2_reg):
        """
        Compute the Wasserstein distance for the discriminator.

        Parameters
        ----------
        orig_output: tf.Tensor
            Output of the discriminator on the original data
        synth_output: tf.Tensor
            Output of the discriminator on the synthetic data
        interp_grad: tf.Tensor
            Gradient of the discriminator at the interpolated data
        l2_reg: tf.Tensor
            Loss for the L2 regularization

        Returns
        -------
        tf.Tensor:
            Final loss function for the discriminator. (to be passed to the optimizer)
        """

        real_loss = tf.reduce_mean(orig_output)
        fake_loss = tf.reduce_mean(synth_output)

        # the gradient penalty loss
        interp_grad_norm = tf.sqrt(tf.reduce_sum(tf.square(interp_grad), axis=[1]))
        grad_pen = tf.multiply(self.lambda_, tf.reduce_mean((interp_grad_norm - 1.0) ** 2))

        # Full loss
        loss = fake_loss - real_loss + grad_pen + l2_reg

        # Log stuff
        self.logs['discriminator']['loss_real'] = real_loss
        self.logs['discriminator']['loss_fake'] = fake_loss
        self.logs['discriminator']['wass_loss'] = fake_loss - real_loss
        self.logs['discriminator']['grad_pen'] = grad_pen
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

        for l in ['loss_real', 'loss_fake', 'wass_loss', 'grad_pen', 'loss', 'reg_loss']:
            logs['discriminator'][l] = []

        logs['generator'] = {}

        for l in ['gen_loss', 'kl_div', 'loss', 'reg_loss']:
            logs['generator'][l] = []
