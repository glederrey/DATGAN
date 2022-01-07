#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import tensorflow as tf

from tensorpack.utils import logger

from datgan.utils.data import Preprocessor


class DATGAN:
    """
    Main class for DATGAN synthesizer.

    Attributes
    ----------
        continuous_columns:  list[str]
            List of variables to be considered continuous.

    Methods
    -------
        preprocessing

        fit

        sample

    """

    def __init__(self, loss_function=None, label_smoothing='TS', output='output', gpu=True, max_epoch=100,
                 batch_size=500, save_checkpoints=True, restore_session=True,  learning_rate=None, z_dim=200,
                 num_gen_rnn=100, num_gen_hidden=50, num_dis_layers=1, num_dis_hidden=100, noise=0.2, l2norm=0.00001,
                 ):
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
            max_epoch: int, default 100
                Number of epochs to use during training.
            batch_size: int, default 500
                Size of the batch to feed the model at each step.
            save_checkpoints: bool, default True
                Whether or not to store checkpoints of the model after each training epoch.
            restore_session: bool, default True
                Whether or not continue training from the last checkpoint.
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

            Raises
            ------
                ValueError
                    If the parameter loss_function is not correctly defined.
                ValueError
                    If the parameter label_smoothing is not correctly defined.
        """

        self.output = output

        # Make sure this directory exists
        self.log_dir = os.path.join(self.output, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Make sure this directory exists
        self.model_dir = os.path.join(self.output, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.restore_path = os.path.join(self.model_dir, 'checkpoint')

        # Training parameters
        self.max_epoch = max_epoch
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
            if tf.test.is_gpu_available() and tf.test.is_built_with_cuda():
                os.environ['CUDA_VISIBLE_DEVICES'] = "0"

        # General variables that are defined in other functions
        self.model = None
        self.trainer = None
        self.preprocessor = None
        self.transformed_data = None
        self.continuous_columns = None

        # Variables for the DAG (defined in the fit function)
        self.dag = None
        self.var_order = None
        self.n_sources = None

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

        # Load the existing preprocessor
        if preprocessed_data_path:
            if os.path.exists(os.path.join(preprocessed_data_path, 'preprocessed_data.pkl')) and \
                    os.path.exists(os.path.join(preprocessed_data_path, 'preprocessor.pkl')):

                # Load the preprocessor and the preprocessed data
                with open(os.path.join(self.data_dir, 'preprocessed_data.pkl'), 'rb') as f:
                    self.transformed_data = pickle.load(f)
                with open(os.path.join(self.data_dir, 'preprocessor.pkl'), 'rb') as f:
                    self.preprocessor = pickle.load(f)

                logger.info("Preprocessed data have been loaded!")
            else:
                raise FileNotFoundError("Could not find the needed files in the folder {}. Make sure the files "
                                        "'preprocessed_data.pkl' and 'preprocessor.pkl' exists."
                                        .format(preprocessed_data_path))

        else:

            logger.info("Preprocessing the data!")

            preprocessed_data_path = os.path.join(self.output, 'encoded_data')
            if not os.path.exists(preprocessed_data_path):
                os.makedirs(preprocessed_data_path)

            self.preprocessor = Preprocessor(continuous_columns=continuous_columns)
            self.transformed_data = self.preprocessor.fit_transform(data)

            # Save them both
            with open(os.path.join(preprocessed_data_path, 'preprocessed_data.pkl'), 'wb') as f:
                pickle.dump(self.transformed_data, f)
            with open(os.path.join(preprocessed_data_path, 'preprocessor.pkl'), 'wb') as f:
                pickle.dump(self.preprocessor, f)

            # Verification for continuous mixture
            self.preprocessor.plot_continuous_mixtures(data, preprocessed_data_path)

            logger.info("Preprocessed data have been saved in '{}'".format(preprocessed_data_path))

    def fit(self, df, continuous_columns):
        """

        Parameters
        ----------
            continuous_columns: list[str]
                List of variables to be considered continuous.

        """


        print('TEST')

    def sample(self, n_rows):
        """

        Parameters
        ----------
            n_rows

        Returns
        -------

        """

        print('TEST')
