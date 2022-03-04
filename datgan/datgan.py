#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import dill
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

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

    def __init__(self, loss_function=None, label_smoothing='TS', output='output', gpu=None, num_epochs=100,
                 batch_size=500, save_checkpoints=True, restore_session=True, learning_rate=None, g_period=None,
                 l2_reg=None, z_dim=200, num_gen_rnn=100, num_gen_hidden=50, num_dis_layers=1, num_dis_hidden=100,
                 noise=0.2, verbose=1):
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
            gpu: int, default None
                Model will automatically try to use GPU if tensorflow can use CUDA. However, this parameter allows you
                to choose which GPU you want to use.
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
            g_period: int, default None
                Every "g_period" steps, train the generator once. (Used to train the discriminator more than the
                generator) By default, it will choose values according the chosen loss function.
            l2_reg: bool, default None
                Tell the model to use L2 regularization while training both NNs. By default, it applies the L2
                regularization when using the SGAN loss function.
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
        self.learning_rate = learning_rate
        self.g_period = g_period
        self.l2_reg = l2_reg
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
        if len(tf.config.list_physical_devices('GPU')) > 0 and tf.test.is_built_with_cuda():
            os.environ['CUDA_VISIBLE_DEVICES'] = "0" if not gpu else str(gpu)

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

    def preprocess(self, data, metadata, preprocessed_data_path=None):
        """
        Preprocess the original data to transform it into a usable dataset.

        Parameters
        ----------
        data: pandas.DataFrame
            Original dataset
        metadata: dict
            Dictionary containing information about the data in the dataframe
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
                self.encoded_data = dill.load(f)

            if self.verbose > 0:
                print("Preprocessed data have been loaded!")
        else:
            if self.verbose > 0:
                print("Preprocessing the data!")

            # Preprocess the original data
            self.encoded_data = EncodedDataset(data, metadata, self.verbose)
            self.encoded_data.fit_transform(fitting=True)

            # Save them both
            with open(os.path.join(self.data_dir, 'encoded_data.pkl'), 'wb') as f:
                dill.dump(self.encoded_data, f)

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

    def fit(self, data, metadata=None, dag=None, preprocessed_data_path=None):
        """
        Fit the DATGAN model to the original data and save it once it's finished training.
        Parameters
        ----------
        data: pandas.DataFrame
            Original dataset
        metadata: dict, default None
            Dictionary containing information about the data in the dataframe
        dag: networkx.DiGraph, default None
            Directed Acyclic Graph representing the relations between the variables. If no dag is provided, the
            algorithm will create a linear DAG.
        preprocessed_data_path: str, default None
            Path to an existing preprocessor. If None is given, the model will preprocess the data and save it under
            self.output + '/encoded_data'.
        """

        # Preprocess the original data
        self.preprocess(data, metadata, preprocessed_data_path)
        self.metadata = self.encoded_data.metadata

        # Verify the integrity of the DAG and get the ordered list of variables for the Generator
        if not dag:
            self.dag = linear_DAG(data)
        else:
            self.dag = dag

        verify_dag(data, dag)
        self.var_order, self.n_sources = get_order_variables(dag)

        self.default_parameter_values(data)

        # Create the folders used to save the checkpoints of the model
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        start = datetime.now()
        start_t = time.perf_counter()

        # Load a new synthesizer
        if self.verbose > 0:
            dt_string = start.strftime("%d/%m/%Y %H:%M:%S")
            print("Start training DATGAN with the {} loss ({}).".format(self.loss_function, dt_string))

        self.synthesizer = Synthesizer(self.output, self.metadata, self.dag, self.batch_size, self.z_dim,
                                       self.noise, self.learning_rate, self.g_period, self.l2_reg, self.num_gen_rnn,
                                       self.num_gen_hidden, self.num_dis_layers, self.num_dis_hidden,
                                       self.label_smoothing, self.loss_function, self.var_order, self.n_sources,
                                       self.save_checkpoints, self.restore_session, self.verbose)

        # Fit the model
        self.synthesizer.fit(self.encoded_data.data, self.num_epochs)

        end = datetime.now()
        if self.verbose > 0:
            dt_string = end.strftime("%d/%m/%Y %H:%M:%S")

            delta = time.perf_counter()-start_t
            str_delta = elapsed_time(delta)

            print("DATGAN has finished training ({}) - Training time: {}".format(dt_string, str_delta))

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                                  Sampling                                                  """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def sample(self, num_samples, sampling='SS'):
        """
        Create a DataFrame with the synthetic generate data
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

        self.encoded_data.set_sampling_technique(sampling)

        num_sampled_data = 0
        final_samples = pd.DataFrame()

        while num_sampled_data <= num_samples:

            encoded_samples = self.synthesizer.sample(self.batch_size)

            decoded_samples = self.encoded_data.reverse_transform(encoded_samples).copy()

            idx_to_keep = np.ones(self.batch_size, dtype=bool)

            for col in self.metadata['details'].keys():
                col_details = self.metadata['details'][col]

                if col_details['type'] == 'continuous':
                    # Check if we need to transform in a discrete distribution
                    if col_details['discrete']:
                        decoded_samples[col] = np.round(decoded_samples[col])

                    # Check the bounds
                    idx = (decoded_samples[col] >= col_details['bounds'][0]) & \
                          (decoded_samples[col] <= col_details['bounds'][1])

                    idx_to_keep *= np.array(idx)

            # Remove the values that are outside of the bounds
            decoded_samples = decoded_samples[idx_to_keep]

            num_sampled_data += len(decoded_samples)

            final_samples = pd.concat([final_samples, decoded_samples], ignore_index=True)

        # Now the df is too big => we make sure it has the right size
        final_samples = final_samples.sample(num_samples)
        final_samples.index = range(len(final_samples))
        return final_samples

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                              Load the model                                                """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def load(self, data, dag=None, preprocessed_data_path=None):
        """
        Load the model based on the latest checkpoint

        Parameters
        ----------
        data: pandas.DataFrame
            Original dataset
        dag: networkx.DiGraph, default None
            Directed Acyclic Graph representing the relations between the variables. If no dag is provided, the
            algorithm will create a linear DAG.
        preprocessed_data_path: str, default None
            Path to an existing preprocessor. If None is given, the model will preprocess the data and save it under
            self.output + '/encoded_data'.
        """
        # Preprocess the original data
        self.preprocess(data, None, preprocessed_data_path)
        self.metadata = self.encoded_data.metadata

        # Reload the DAG
        if not dag:
            self.dag = linear_DAG(data)
        else:
            self.dag = dag

        self.var_order, self.n_sources = get_order_variables(dag)

        self.synthesizer = Synthesizer(self.output, self.metadata, self.dag, self.batch_size, self.z_dim,
                                       self.noise, self.learning_rate, self.g_period, self.l2_reg, self.num_gen_rnn,
                                       self.num_gen_hidden, self.num_dis_layers, self.num_dis_hidden,
                                       self.label_smoothing, self.loss_function, self.var_order, self.n_sources,
                                       self.save_checkpoints, self.restore_session, self.verbose)

        self.synthesizer.initialize()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                              Other functions                                               """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def default_parameter_values(self, data):
        """
        Define some basic parameters based on the chosen loss function

        Parameters
        ----------
        data: pandas.DataFrame
            Original dataset
        """

        # Define the loss function if not defined yet
        if not self.loss_function:

            nbr_cont_cols = 0
            for col in self.metadata['details'].keys():
                if self.metadata['details'][col]['type'] == 'continuous':
                    nbr_cont_cols += 1

            # more categorical columns than continuous
            if float(nbr_cont_cols) / float(len(data.columns)) < 0.5:
                self.loss_function = 'WGAN'
            else:
                self.loss_function = 'WGGP'

        # Define the learning rate
        if not self.learning_rate:
            if self.loss_function == 'SGAN':
                self.learning_rate = 1e-3
            elif self.loss_function == 'WGAN':
                self.learning_rate = 2e-4
            elif self.loss_function == 'WGGP':
                self.learning_rate = 1e-4

        # Define the g_period
        if not self.g_period:
            if self.loss_function == 'SGAN':
                self.g_period = 1
            elif self.loss_function == 'WGAN':
                self.g_period = 2
            elif self.loss_function == 'WGGP':
                self.g_period = 5

        # Define the l2 regularization
        if not self.l2_reg:
            if self.loss_function == 'SGAN':
                self.l2_reg = True
            else:
                self.l2_reg = False


