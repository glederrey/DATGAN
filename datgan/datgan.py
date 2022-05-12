#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import dill
import types
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from datgan.utils.utils import elapsed_time
from datgan.utils.data import EncodedDataset
from datgan.synthesizer.synthesizer import Synthesizer
from datgan.utils.dag import verify_dag, get_order_variables, linear_dag, transform_dag


class DATGAN:
    """
    The DATGAN is a synthesizer for tabular data. It uses LSTM cells to generate synthetic data for continuous and
    categorical variable types. In addition, a Directed Acyclic Graph (DAG) can be provided to represent the structure
    between the variables and help the model to perform better. This model integrates two types of conditionality:
    rejection by sampling and conditional inputs.

    """

    def __init__(self, loss_function=None, label_smoothing='TS', output='output', gpu=None, num_epochs=100,
                 batch_size=500, save_checkpoints=True, restore_session=True, learning_rate=None, g_period=None,
                 l2_reg=None, z_dim=200, num_gen_rnn=100, num_gen_hidden=50, num_dis_layers=1, num_dis_hidden=100,
                 noise=0.5, conditional_inputs=None, verbose=1):
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
        conditional_inputs: list, default None
            List of variables in the dataset that are used as conditional inputs to the model.
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
        self.conditional_inputs = conditional_inputs if conditional_inputs else []
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
        Preprocess the original data to transform it into a usable dataset for the DATGAN model.

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
        """

        # Check that the conditional inputs corresponds to some variables in the dataset
        for c in self.conditional_inputs:
            if c not in data.columns:
                raise ValueError("The conditional input '{}' does not appear in the column names of the dataset ({})."
                                 .format(c, list(data.columns)))

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
            self.encoded_data.fit_transform()

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
        Fit the DATGAN model to the original encoded data.

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

        if not dag:
            self.dag = linear_dag(data, self.conditional_inputs)
        else:
            self.dag = dag

        # Transform the DAG depending on the conditional inputs
        self.dag = transform_dag(self.dag, self.conditional_inputs)

        # Verify the integrity of the DAG and get the ordered list of variables for the Generator
        verify_dag(data, self.dag)
        self.var_order, self.n_sources = get_order_variables(self.dag)

        self.__default_parameter_values(data)

        # Create the folders used to save the checkpoints of the model
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        start = datetime.now()
        start_t = time.perf_counter()

        # Load a new synthesizer
        if self.verbose > 0:
            dt_string = start.strftime("%d/%m/%Y %H:%M:%S")
            print("Start training DATGAN with the {} loss ({}).".format(self.loss_function, dt_string))

        self.synthesizer = Synthesizer(self.output, self.metadata, self.dag, self.batch_size, self.z_dim, self.noise,
                                       self.learning_rate, self.g_period, self.l2_reg, self.num_gen_rnn,
                                       self.num_gen_hidden, self.num_dis_layers, self.num_dis_hidden,
                                       self.label_smoothing, self.loss_function, self.var_order, self.n_sources,
                                       self.conditional_inputs, self.save_checkpoints, self.restore_session,
                                       self.verbose)

        # Fit the model
        self.synthesizer.fit(self.encoded_data.data, self.num_epochs)

        end = datetime.now()
        if self.verbose > 0:
            dt_string = end.strftime("%d/%m/%Y %H:%M:%S")

            delta = time.perf_counter() - start_t
            str_delta = elapsed_time(delta)

            print("DATGAN has finished training ({}) - Training time: {}".format(dt_string, str_delta))

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                                  Sampling                                                  """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def sample(self, num_samples, inputs=None, cond_dict=None, sampling='SS', randomize=True, timeout=True):
        """
        Create a DataFrame with the synthetic generate data. Conditionality is done through a rejection sampling
        process. For categorical variables, you need to provide the categories you want to get. For continuous
        variables, you need to provide a lambda function that returns a boolean based on the values of the variable.

        Parameters
        ----------
        num_samples: int
            Number of sample to provide
        inputs: dict or pandas.DataFrame, default None
            Optional input data. Used if the conditional inputs have been set.
        cond_dict: dict, default None
            Conditional dictionary.
        sampling: str, default 'SS'
            Type of sampling to use. Only accepts the following values: 'SS', 'SA', 'AS', and 'AA'
        randomize: bool, default True
            Randomize the conditional inputs if set to True. If set to False, it will not discard any samples.
        timeout: bool, default True
            Use a timeout to stop sampling if the model can't generate the data asked in the conditional dict

        Returns
        -------
        pandas.DataFrame:
            Synthetic dataset of 'num_samples' rows
        """

        if cond_dict is None:
            cond_dict = {}

        # Set the sampling technique (simulation and/or argmax)
        self.encoded_data.set_sampling_technique(sampling)

        # Prepare the conditional inputs if needed
        if len(self.conditional_inputs) > 0:
            prep_inputs = self.__prepare_inputs(inputs)
            len_inputs = len(inputs[self.conditional_inputs[0]])
            idx_inputs = list(range(len_inputs))
        else:
            prep_inputs = {}
            len_inputs = 0
            idx_inputs = None

        # Test that the conditional dict for the rejection sampling is correct
        self.__test_conditional_dict(cond_dict)

        # Prepare some variables
        num_sampled_data = 0
        samples = pd.DataFrame()
        count_no_samples = 0

        if self.verbose > 0:
            pbar = tqdm(total=num_samples, desc="Sampling from DATGAN")

        if not randomize and num_samples != len_inputs:
            num_samples = len_inputs
            if self.verbose > 0:
                print("Number of samples have been adjusted to the size of the conditional inputs.")

        if not randomize:
            set_idx = set(range(len_inputs))

        # While loop until we have enough samples
        while num_sampled_data < num_samples:

            if len(self.conditional_inputs) > 0:
                # Select randomly values in the cond_inputs dictionary
                if randomize:
                    samp_idx = np.random.choice(idx_inputs, self.batch_size, replace=(len_inputs < self.batch_size))
                else:
                    if len(set_idx) >= self.batch_size:
                        samp_idx = np.random.choice(list(set_idx), self.batch_size, replace=False)
                    else:
                        samp_idx = np.concatenate([list(set_idx), np.zeros(self.batch_size - len(set_idx))])
                        samp_idx = samp_idx.astype(int)

                samples_inputs = {}
                for c in prep_inputs:
                    samples_inputs[c] = tf.gather(prep_inputs[c], samp_idx, axis=0)
            else:
                samp_idx = None
                samples_inputs = None

            # Get samples from the synthesizer
            encoded_samples = self.synthesizer.sample(samples_inputs)

            # Decode the data
            synth_samples = self.encoded_data.reverse_transform(encoded_samples).copy()

            # Replace the conditional inputs values in the decoded samples by the original values
            for col in self.conditional_inputs:
                synth_samples[col] = list(inputs[col][samp_idx])

            # Check the values of the samples
            idx_to_keep = self.__review_sampled_data(synth_samples, cond_dict)

            # Remove all the excess samples
            if not randomize:
                if len(set_idx) < self.batch_size:
                    idx_to_keep[len(set_idx):] = False

            # Select a subset of the samples according to the conditional dictionary
            synth_samples = synth_samples[idx_to_keep]

            # Update the set of index to be sampled
            if not randomize:
                idx_kept = samp_idx[idx_to_keep]
                synth_samples['index'] = idx_kept
                set_idx = set_idx - set(idx_kept)

            n_samp = len(synth_samples)

            # Check if we could sample some synthetic data or not
            if n_samp > 0:
                count_no_samples = 0
            else:
                count_no_samples += 1

            # If timeout, we stop sampling now
            if timeout and count_no_samples == 3:
                raise TimeoutError("DATGAN was not able to provide any samples with the required copnditionals 3 "
                                   "times in a row. => Sampling is stopped.")

            # Update the progress bar
            if self.verbose > 0:
                if num_sampled_data + n_samp >= num_samples:
                    pbar.update(num_samples - num_sampled_data)
                else:
                    pbar.update(n_samp)

            num_sampled_data += n_samp

            # Add the current samples to the final DataFrame
            samples = pd.concat([samples, synth_samples], ignore_index=True)

        if randomize:
            # Now the df is too big => we make sure it has the right size
            samples = samples.sample(num_samples)
        else:
            samples = samples.sort_values('index')

        samples.index = range(len(samples))

        if self.verbose > 0:
            pbar.close()

        return samples.reindex(self.encoded_data.original_columns, axis=1)

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
            self.dag = linear_dag(data, self.conditional_inputs)
        else:
            self.dag = dag

        # Transform the DAG depending on the conditional inputs
        self.dag = transform_dag(dag, self.conditional_inputs)

        self.var_order, self.n_sources = get_order_variables(dag)

        self.synthesizer = Synthesizer(self.output, self.metadata, self.dag, self.batch_size, self.z_dim, self.noise,
                                       self.learning_rate, self.g_period, self.l2_reg, self.num_gen_rnn,
                                       self.num_gen_hidden, self.num_dis_layers, self.num_dis_hidden,
                                       self.label_smoothing, self.loss_function, self.var_order, self.n_sources,
                                       self.conditional_inputs, self.save_checkpoints, self.restore_session,
                                       self.verbose)

        self.synthesizer.initialize()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                              Private methods                                               """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def __default_parameter_values(self, data):
        """
        Define some basic parameters based on the chosen loss function. Used in the function `fit`.

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

    def __prepare_inputs(self, inputs):
        """
        Prepare the inputs by transforming them according the preprocessed data. The encoded values will then be
        passed to the DATGAN for sampling synthetic data.

        Parameters
        ----------
        inputs: dict
            Dictionary with the inputs used as conditionals

        Returns
        -------
        prep_inputs: dict
            Transformed values of inputs
        """

        prep_inputs = {}
        if inputs is None:
            raise ValueError("You need to provide the conditional inputs.")
        else:
            # We need to transform the inputs according the transformation done in the encoding process
            for col in inputs:
                col_details = self.metadata['details'][col]

                if col_details['type'] == 'continuous':
                    data = np.array(inputs[col])

                    # Apply lambda function if provided
                    if 'apply_func' in col_details:
                        data = col_details['apply_func'](data)

                    data = data.reshape(-1, 1)

                    # Transform the provided inputs using the GMM
                    model = col_details['transform']
                    n_modes = col_details['n']

                    means = model.means_.reshape((1, n_modes))
                    stds = np.sqrt(model.covariances_).reshape((1, n_modes))

                    # Normalization
                    normalized_values = ((data - means) / (self.encoded_data.continuous_transformer.std_span * stds))
                    probs = model.predict_proba(data)

                    # Clip the values
                    normalized_values = np.clip(normalized_values, -.99, .99)

                    prep_inputs[col] = tf.convert_to_tensor(
                        np.concatenate([normalized_values, probs], axis=1), dtype=tf.float32)
                elif col_details['type'] == 'categorical':
                    # We need to encode the labels from "str" to "int"
                    cat_encoder = LabelEncoder()
                    cat_encoder.classes_ = col_details['mapping']
                    prep_inputs[col] = tf.one_hot(cat_encoder.transform(inputs[col].astype(str)),
                                                  depth=len(col_details['mapping']))

        return prep_inputs

    def __test_conditional_dict(self, cond_dict):
        """
        Test that the values provided in the conditional dictionary can be used to sample the DATGAN.

        Parameters
        ----------
        cond_dict: dict, default None
            Conditional dictionary.

        Raises
        -------
        ValueError if the values do not correspond to what is expected.
        """

        # Test the dictionary for conditionals
        for k in cond_dict.keys():
            col_details = self.metadata['details'][k]

            if col_details['type'] == 'categorical':

                # Check that the type is ok
                type_ok = False
                if isinstance(cond_dict[k], str):
                    type_ok = True

                    # Check that the value exists in the possible categories
                    if cond_dict[k] not in col_details['mapping']:
                        raise ValueError("The key {} for the variable {} does not correspond to an existing category "
                                         "({})".format(cond_dict[k], k, col_details['mapping']))

                    # Transform into a list
                    cond_dict[k] = [cond_dict[k]]

                elif isinstance(cond_dict[k], list):
                    type_ok = True
                    for el in cond_dict[k]:
                        if not isinstance(el, str):
                            type_ok = False

                        # Check that the value exists in the possible categories
                        if el not in col_details['mapping']:
                            raise ValueError(
                                "The key '{}' for the variable '{}' does not correspond to an existing category "
                                "({})".format(el, k, col_details['mapping']))

                if not type_ok:
                    raise ValueError("The values in the conditional dictionary for the categorical variable '{}' must "
                                     "be of type 'str' or 'list' of 'str'. You have given the following value(s): {}"
                                     .format(k, cond_dict[k]))

            else:
                if not isinstance(cond_dict[k], types.LambdaType):
                    raise ValueError("The value for the continuous variable '{}' has to be a lambda function "
                                     "that returns a boolean value!".format(k))

                test = cond_dict[k](0)

                if not isinstance(test, bool):
                    raise ValueError("The lambda function provided for the continuous variable '{}' must return a "
                                     "boolean value!".format(k))

    def __review_sampled_data(self, synth_samples, cond_dict):
        """
        Discard sampled data based on the bounds (or enforce the bounds) and/or the conditional dictionary.

        => Core of the rejection sampling process.

        Parameters
        ----------
        synth_samples: pandas.DataFrame
            DataFrame of the synthetic data
        cond_dict: dict
            Conditional dictionary

        Returns
        -------
        idx_to_keep: list[int]
            List of 0 and 1s used to keep or discard values in the `synth_samples` DataFrame.
        """

        idx_to_keep = np.ones(self.batch_size, dtype=bool)

        # Go through each columns
        for col in self.metadata['details'].keys():
            col_details = self.metadata['details'][col]

            if col_details['type'] == 'continuous':
                # Check if we need to transform in a discrete distribution
                if col_details['discrete']:
                    synth_samples[col] = np.round(synth_samples[col])

                # Check the bounds
                if 'bounds' in col_details.keys():
                    if col_details['enforce_bounds']:
                        synth_samples[col] = synth_samples[col].clip(col_details['bounds'][0],
                                                                     col_details['bounds'][1])
                    else:
                        idx = (synth_samples[col] >= col_details['bounds'][0]) & \
                              (synth_samples[col] <= col_details['bounds'][1])

                        idx_to_keep *= np.array(idx)

            # Check if the column is in the conditional dictionary
            if col in cond_dict.keys():
                if col_details['type'] == 'continuous':
                    # Apply the lambda function returning boolean values
                    idx = cond_dict[col](synth_samples[col])
                else:
                    idx = np.zeros(self.batch_size, dtype=bool)

                    # Check if the sampled values is equal to one value in the list of the conditional dictionary
                    for k in cond_dict[col]:
                        idx += (synth_samples[col] == k)

                    # Replace values greater than 1
                    idx[idx > 1] = 1

                idx_to_keep *= np.array(idx)

        return idx_to_keep
