#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf


class DATGAN:
    """
    Main class for DATGAN synthesizer.

    Attributes:
        continuous_columns (list[str]): List of variables to be considered continuous.

    """

    def __init__(self, continuous_columns, output='output', gpu=True, max_epoch=100,
                 label_smoothing='TS', noise=0.2, save_checkpoints=True, restore_session=True,
                 batch_size=500, z_dim=200, l2norm=0.00001, learning_rate=None, num_gen_rnn=100,
                 num_gen_hidden=50, num_dis_layers=1, num_dis_hidden=100
                 ):
        """
        Constructs all the necessary attributes for the DATGAN class.

        Args:
            continuous_columns (list[str]): List of variables to be considered continuous.
            output (str, optional): Path to store the model and its artifacts. Defaults to 'output'.
            gpu (bool, optional): Use the first available GPU. Defaults to 'True'.
            max_epoch (int, optional): Number of epochs to use during training. Defaults to '5'.
            label_smoothing (str, optional): Type of label smoothing. Only accepts the values 'TS', 'OS', and 'NO'.
                Defaults to 'TS'.
            noise (float, optional): Upper bound to the gaussian noise added to with the label smoothing. Defaults to
                `0.2`.
            save_checkpoints(bool, optional): Whether or not to store checkpoints of the model after each training
                epoch. Defaults to `True`
            restore_session(bool, optional): Whether or not continue training from the last checkpoint. Defaults to
                `True`.
            batch_size (int, optional): Size of the batch to feed the model at each step. Defaults to `500`.
            z_dim (int, optional): Number of dimensions in the noise input for the generator. Defaults to :attr:`100`.
            l2norm (float, optional): L2 reguralization coefficient when computing the standard GAN loss. Defaults to
                `0.00001`.
            learning_rate (float, optional): Learning rate for the optimizer (depends on the chosen loss function).
                Defaults to 'None`.
            num_gen_rnn (int, optional): Size of the hidden units in the LSTM cell. Defaults to `100`.
            num_gen_hidden (int, optional): Size of the hidden layer used on the output of the generator. Defaults to
                `50`.
            num_dis_layers (int, optional): Number of layers for the discriminator. Defaults to `1`.
            num_dis_hidden (int, optional): Size of the hidden layers in the discriminator. Defaults to `100`.
        """

        self.continuous_columns = continuous_columns

        # Preprocessing takes care of this directory
        self.data_dir = os.path.join(output, 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Make sure this directory exists
        self.log_dir = os.path.join(output, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Make sure this directory exists
        self.model_dir = os.path.join(output, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Variables for the DAG
        self.dag = None
        self.var_order = None
        self.n_sources = None

        # Training parameters
        self.max_epoch = max_epoch
        self.save_checkpoints = save_checkpoints
        self.restore_session = restore_session

        # Model parameters
        self.model = None
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.l2norm = l2norm
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_hidden = num_gen_hidden
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden

        # Specific parameters for the DATGAN
        self.label_smoothing = label_smoothing
        if self.label_smoothing not in ['TS', 'OS', 'NO']:
            raise ValueError("The variable 'label_smoothing' must take one of the following values: 'TS' (two-sided), "
                             "'OS' (one-sided), or 'NO' (none)")

        # Check if there's a GPU available and tensorflow has been compiled with cuda
        if gpu:
            if tf.test.is_gpu_available() and tf.test.is_built_with_cuda():
                os.environ['CUDA_VISIBLE_DEVICES'] = "0"
