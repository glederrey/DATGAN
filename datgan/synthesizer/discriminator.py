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
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))]
            else:
                internal = [layers.Dense(self.num_dis_hidden)]

            # Add the layer for the batch_diversity
            internal.append(layers.Dense(self.n_kernel*self.kernel_dim))

            if self.loss_function == 'WGGP':
                internal.append(layers.LayerNormalization())
            else:
                internal.append(layers.BatchNormalization())

            internal.append(layers.Dropout(0.5))
            internal.append(layers.LeakyReLU())

            self.list_layers.append(internal)

        if self.loss_function == 'SGAN':
            self.list_layers.append(layers.Dense(1, activation='softmax'))
        else:
            self.list_layers.append(layers.Dense(1))

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

            # Pass through Dropout
            x = internal[3](x)

            # Use leaky_ReLU
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
