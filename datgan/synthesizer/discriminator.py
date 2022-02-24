#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File that describes the Discriminator for the DATGAN
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


class Discriminator(keras.Model):
    """
    The discriminator is a fully connected neural network. Exactly the same as in TGAN. The only difference is that
    Layer Normalization is used instead of Batch Normalization with the 'WGGP' loss.

    """

    def __init__(self, num_dis_layers, num_dis_hidden, loss_function):
        """
        Initialize the model

        Parameters
        ----------
        num_dis_layers: int
            Number of layers for the discriminator. (Default value in class DATGAN: 1)
        num_dis_hidden: int
            Size of the hidden layers in the discriminator. (Default value in class DATGAN: 100)
        loss_function: str
            Name of the loss function to be used. (Defined in the class DATGAN)
        """
        super().__init__()
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden
        self.loss_function = loss_function

        # Batch diversity parameters
        self.n_kernel = 10
        self.kernel_dim = 10

        # Regularizer (used with the SGAN loss function)
        self.kern_reg = None
        if self.loss_function == 'SGAN':
            self.kern_reg = tf.keras.regularizers.L2(1e-5)

        # Constraints (used with the WGAN loss function)
        self.constraints = None
        if self.loss_function == 'WGAN':
            self.constraints = MinMaxConstraint(x_min=-0.01, x_max=0.01)

        self.list_layers = None
        self.build_layers()

    def build_layers(self):
        """
        Build the layers of the Discriminator.
        """

        self.list_layers = []

        for i in range(self.num_dis_layers):

            if i == 0:
                internal = [layers.Dense(self.num_dis_hidden,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                         kernel_regularizer=self.kern_reg,
                                         kernel_constraint=self.constraints,
                                         bias_constraint=self.constraints)]
            else:
                internal = [layers.Dense(self.num_dis_hidden,
                                         kernel_regularizer=self.kern_reg,
                                         kernel_constraint=self.constraints,
                                         bias_constraint=self.constraints)]

            # Add the layer for the batch_diversity
            internal.append(layers.Dense(self.n_kernel*self.kernel_dim,
                                         kernel_regularizer=self.kern_reg,
                                         kernel_constraint=self.constraints,
                                         bias_constraint=self.constraints))

            # No need to use the scale parameters for the normalization since the results will be passed to the Dropout
            # and LeakyReLU layers
            if self.loss_function == 'WGGP':
                internal.append(layers.LayerNormalization(center=True,
                                                          scale=False,
                                                          beta_regularizer=self.kern_reg,
                                                          beta_constraint=self.constraints))
            else:
                # Don't use the gamma parameters in the BatchNormalization
                internal.append(layers.BatchNormalization(center=True,
                                                          scale=False,
                                                          beta_regularizer=self.kern_reg,
                                                          beta_constraint=self.constraints))

            internal.append(layers.LeakyReLU())

            internal.append(layers.Dropout(0.2))

            self.list_layers.append(internal)

        if self.loss_function == 'SGAN':
            self.list_layers.append(layers.Dense(1,
                                                 activation='sigmoid',
                                                 kernel_regularizer=self.kern_reg,
                                                 kernel_constraint=self.constraints,
                                                 bias_constraint=self.constraints))
        else:
            self.list_layers.append(layers.Dense(1,
                                                 kernel_regularizer=self.kern_reg,
                                                 kernel_constraint=self.constraints,
                                                 bias_constraint=self.constraints))

    def call(self, x):
        """
        Compute the Discriminator value

        Parameters
        ----------
        x: torch.Tensor
            A Torch Tensor of dimensions (N, n_features)

        Returns
        -------
        torch.Tensor:
            Critic of the current input of dimensions (N, 1)
        """

        for i in range(self.num_dis_layers):
            internal = self.list_layers[i]

            # Initial layer
            x = internal[0](x)

            # Concatenation with batch diversity
            x = tf.concat([x, self.batch_diversity(internal[1](x))], axis=1)

            # Pass through LayerNorm or BatchNorm
            x = internal[2](x)

            # Use leaky_ReLU
            x = internal[3](x)

            # Pass through Dropout
            x = internal[4](x)

        # Pass through the output layer
        return self.list_layers[-1](x)

    def batch_diversity(self, M):
        """
        Return the minibatch discrimination vector as defined by Salimans et al., 2016.


        Parameters
        ----------
        M: tf.keras.layers.Dense
            Input layer

        Returns
        -------
        tensorflow.Tensor:
            batch diversity tensor

        """
        M = tf.reshape(M, [-1, self.n_kernel, self.kernel_dim])
        M1 = tf.reshape(M, [-1, 1, self.n_kernel, self.kernel_dim])
        M2 = tf.reshape(M, [1, -1, self.n_kernel, self.kernel_dim])
        diff = tf.exp(-tf.reduce_sum(tf.abs(M1 - M2), axis=3))
        return tf.reduce_sum(diff, axis=0)


class MinMaxConstraint(tf.keras.constraints.Constraint):
    """constrain model weights between [x_min, x_max]."""

    def __init__(self, x_min=0.0, x_max=1.0):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        # TODO: add sanity check if x_max == x_min or w_max == w_min

    def __call__(self, w):
        w_min = tf.minimum(tf.math.reduce_min(w), self.x_min)
        w_max = tf.maximum(tf.math.reduce_max(w), self.x_max)
        scale = (self.x_max - self.x_min) / (w_max - w_min)
        m = self.x_min - w_min * scale
        w = w * scale
        return w + m
