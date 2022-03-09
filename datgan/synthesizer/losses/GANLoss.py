#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class GANLoss(tf.keras.losses.Loss):
    """
    Generic class for the loss function
    """

    def __init__(self, metadata, var_order, conditionality, name="GENERIC_CLASS"):
        """
        Initialize the class

        Parameters
        ----------
        metadata: dict
            Metadata for the columns (used when computing the kl divergence)
        var_order: list
            Ordered list of the variables
        conditionality: bool
            Whether to use conditionality or not
        name: string
            Name of the loss function
        """
        super().__init__(name=name)

        self.logs = {}
        self.reset_logs()

        self.metadata = metadata
        self.var_order = var_order
        self.conditionality = conditionality

        self.kl = tf.keras.losses.KLDivergence()

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
        kl_div = tf.constant(0.0)
        ptr = tf.constant(0)

        # Go through all variables
        for col in self.var_order:
            # Get info
            col_info = self.metadata['details'][col]

            if col_info['type'] == 'continuous':
                # Skip the value. We only compute the KL on the probability vector
                ptr += tf.constant(col_info['n'])

            pred = tf.reduce_sum(synthetic[:, ptr:ptr + col_info['n']], axis=0)
            pred = pred / tf.reduce_sum(pred)

            real = tf.reduce_sum(original[:, ptr:ptr + col_info['n']], axis=0)
            real = real / tf.reduce_sum(real)

            kl_div += self.kl(pred, real)
            ptr += tf.constant(col_info['n'])

        return kl_div

    def cond_loss(self, synth_output, cond):

        ptr_output = 0
        ptr_cond = 0
        loss = []

        for col in self.var_order:

            col_details = self.metadata['details'][col]
            n = col_details['n']
            n_cat = col_details['n_cat']

            if col_details['type'] == 'categorical':
                y = synth_output[:, ptr_output:ptr_output + n]
                x = cond[:, ptr_cond:ptr_cond + n_cat]

                # Cross entropy loss
                tmp = tf.reduce_mean(-tf.reduce_sum(tf.math.xlogy(x, y)))
                loss.append(tmp)
            else:
                ptr_output += n

            ptr_output += n
            ptr_cond += n_cat

        cond_loss = tf.reduce_sum(tf.stack(loss, axis=0))/tf.cast(tf.shape(synth_output)[0], dtype=tf.float32)

        self.logs['generator']['cond_loss'] = cond_loss

        return cond_loss

    def reset_logs(self):
        """
        Reset the logs of the loss to empty dictionaries
        """
        self.logs = {'discriminator': {}, 'generator': {}}

    def get_logs(self, key):
        """
        Return the logs of the loss
        """
        return self.logs[key]

