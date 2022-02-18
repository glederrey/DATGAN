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
                 verbose):
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
        self.verbose = verbose

        # Parameter used for the WGGP loss function
        self.lambda_ = 10

        # Other parameters
        self.generator = None
        self.discriminator = None
        self.logging = {'generator': {}, 'discriminator': {}}

    def fit(self, encoded_data, num_epochs):

        self.generator = Generator(self.metadata, self.dag, self.batch_size, self.z_dim, self.num_gen_rnn,
                                   self.num_gen_hidden, self.var_order, self.verbose)

        z = tf.random.normal([self.n_sources, self.batch_size, self.z_dim])

        outputs = self.generator(z)

        keys = list(outputs.keys())
        val = outputs[keys[0]]
        for k in keys[1:]:
            val = tf.concat([val, outputs[k]], axis=1)

        print(val)

        self.discriminator = Discriminator(self.num_dis_layers, self.num_dis_hidden, self.loss_function)

        out = self.discriminator(val)
        print(out)

        asd

    def save_model(self, epoch, max_epochs):
        """
        Save the synthesizer to a pickle object. Delete the previous checkpoint
        Parameters
        ----------
        epoch: int
            Current epoch
        max_epochs: int
            Maximum number of epochs for training
        """

        # Final model is saved. If save_checkpoints is enabled, model is saved at each epoch
        if self.save_checkpoints or epoch == max_epochs:
            with open(os.path.join(self.output, 'synthesizer.pkl'), 'wb') as outfile:
                pickle.dump(self, outfile)