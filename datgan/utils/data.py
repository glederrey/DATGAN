#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing related functionalities.

This file contains the tools to preprare the data, from the raw csv files, to the DataFlow objects that are used to
fit the DATGAN model.

This file contains the following classes:
- Preprocessor
- MultiModalNumberTransformer
- DATGANDataFlow
- RandomZData
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture

from tensorpack.utils import logger
from tensorpack import DataFlow, RNGDataFlow


class Preprocessor:
    """
    Transform back and forth human-readable data into DATGAN numerical features.

    Continuous columns are transformed using the class `MultiModalNumberTransformer` while the categorical columns are
    one-hot encoded using the class `sklearn.preprocessing.LabelEncoder`.

    Original class from TGAN.
    """

    def __init__(self, continuous_columns):
        """Initialize object.

        Parameters
        ----------
            continuous_columns: list[str]
                List of names of the continuous variables
        """
        if continuous_columns is None:
            continuous_columns = []

        self.continuous_columns = continuous_columns
        self.columns = None

        self.metadata = None
        self.continuous_transformer = MultiModalNumberTransformer()
        self.categorical_transformer = LabelEncoder()
        self.columns = None
        self.categorical_argmax = None

    def set_sampling_technique(self, sampling):
        """
        Choose which type of sampling will be used for both categorical and continuous columns

        Parameters
        ----------
        sampling: str
            Type of sampling that we want to use

        """

        self.categorical_argmax = (sampling in ['AA', 'SA'])
        self.continuous_transformer.argmax = (sampling in ['AA', 'AS'])

    def fit_transform(self, data, fitting=True):
        """
        Transform human-readable data into DATGAN numerical features.

        Parameters
        ----------
            data: pandas.DataFrame
                Data to transform.
            fitting: bool, default True
                Whether or not to update self.metadata.

        Returns
        -------
            dict
                Transformed data using a dict of lists.
        """
        num_cols = data.shape[1]

        transformed_data = {}
        details = {}

        self.columns = data.columns

        for col in self.columns:
            if col in self.continuous_columns:
                logger.info("Encoding continuous variable \"{}\"...".format(col))

                column_data = data[col].values.reshape([-1, 1])
                features, probs, model, n_modes = self.continuous_transformer.transform(column_data)
                transformed_data[col] = np.concatenate((features, probs), axis=1)

                if fitting:
                    details[col] = {
                        "type": "continuous",
                        "n": n_modes,
                        "transform": model,
                    }
            else:
                logger.info("Encoding categorical variable \"{}\"...".format(col))

                column_data = data[col].astype(str).values
                features = self.categorical_transformer.fit_transform(column_data)
                transformed_data[col] = features.reshape([-1, 1])

                if fitting:
                    mapping = self.categorical_transformer.classes_
                    details[col] = {
                        "type": "category",
                        "mapping": mapping,
                        "n": mapping.shape[0],
                    }

        if fitting:
            metadata = {
                "num_features": num_cols,
                "details": details,
            }
            check_metadata(metadata)
            self.metadata = metadata

        return transformed_data

    def transform(self, data):
        """
        Transform the given dataframe without generating new metadata.

        Parameters
        ----------
            data: pandas.DataFrame
                Data to transform.

        Returns
        -------
            pandas.DataFrame
                Model features
        """
        return self.fit_transform(data, fitting=False)

    def fit(self, data):
        """
        Initialize the internal state of the object using `data`.

        Parameters
        ----------
            data: pandas.DataFrame
                Data to transform.
        """
        self.fit_transform(data)

    def reverse_transform(self, data):
        """
        Transform DATGAN numerical features back into human-readable data.

        Parameters
        ----------
            data: pandas.DataFrame
                Data to reverse-transform.

        Returns
        -------
            pandas.DataFrame
                Human-readable data
        """
        table = []

        for col in self.columns:
            column_data = data[col]
            column_metadata = self.metadata['details'][col]

            if column_metadata['type'] == 'continuous':
                column = self.continuous_transformer.inverse_transform(column_data, column_metadata)

            if column_metadata['type'] == 'category':
                self.categorical_transformer.classes_ = column_metadata['mapping']

                selected_component = select_values(column_data, argmax=self.categorical_argmax)

                column = self.categorical_transformer.inverse_transform(selected_component)

            table.append(column)

        result = pd.DataFrame(dict(enumerate(table)))
        result.columns = self.columns
        return result

    def plot_continuous_mixtures(self, data, path):
        """
        Plot and save the continuous distributions with the mixtures

        Parameters
        ----------
            data: pandas.DataFrame
                Original data
            path: str
                Path of the encoded data folder
        """

        path = os.path.join(path, 'continuous')

        if not os.path.exists(path):
            os.makedirs(path)

        for col in self.continuous_columns:

            details = self.metadata['details'][col]

            gmm = details['transform']

            tmp = data[col]

            fig = plt.figure(figsize=(10, 7))
            plt.hist(tmp, 50, density=True, histtype='stepfilled', alpha=0.4, color='gray')

            x = np.linspace(np.min(tmp), np.max(tmp), 1000)

            logprob = gmm.score_samples(x.reshape(-1, 1))
            responsibilities = gmm.predict_proba(x.reshape(-1, 1))
            pdf = np.exp(logprob)
            pdf_individual = responsibilities * pdf[:, np.newaxis]
            plt.plot(x, pdf, '-k')
            plt.plot(x, pdf_individual, '--k')

            plt.xlabel('$x$')
            plt.ylabel('$p(x)$')
            plt.title("{} - {} mixtures".format(col, details['n']))
            plt.savefig(path + '/{}.png'.format(col), bbox_inches='tight', facecolor='white')
            plt.close(fig)


class MultiModalNumberTransformer:
    """
    Reversible transform for multimodal data.

    A Variational Gaussian Mixture (VGM) is trained on a vector of continuous values. The number of modes is adjusted
    based on the results of the trained VGM.

    Original class from TGAN.
    """

    def __init__(self, argmax=True):
        """
        Initialize object.

        Parameters
        ----------
            argmax: bool, default True
                Whether to use argmax or simulation to draw the values from the GMM
        """
        self.max_clusters = 10
        self.std_span = 2
        self.n_bins = 50
        self.thresh = 1e-3
        self.argmax = argmax

        # Remove Convergence warning
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

    def transform(self, data):
        """
        Cluster values using a `sklearn.mixture.BayesianGaussianMixture` model.

        Parameters
        ----------
            data: numpy.ndarray
                Values to cluster in array of shape (n,1).

        Returns
        -------
            tuple: [numpy.ndarray, numpy.ndarray, sklearn.mixture.BayesianGaussianMixture, int]
                Tuple containing the normalized values, probabilities, the VGM model and the number of modes
        """
        n_modes = 10
        logger.info("  Fitting model with {:d} components".format(n_modes))

        while True:

            # Fit the BGM
            model = BayesianGaussianMixture(
                n_components=n_modes,
                max_iter=200,
                n_init=10,
                init_params='kmeans',
                weight_concentration_prior_type='dirichlet_process')

            # Test with less data
            idx = np.random.choice(len(data), min(10000, len(data)))
            samples = data[idx]

            model.fit(samples)

            # Check that BGM is using all the classes!
            pred_ = np.unique(model.predict(samples))

            # Check that the weights are large enough
            w = model.weights_ > 1e-2

            if len(pred_) != n_modes:
                n_modes = len(pred_)
                logger.info("  Predictions were done on {:d} components => Fit with {:d} components!"
                            .format(n_modes, n_modes))
            elif np.sum(w) != n_modes:
                n_modes = np.sum(w)
                logger.info("  Some weights are too small =>  => Fit with {:d} components!".format(n_modes))
            else:
                logger.info("  Predictions were done on {:d} components => FINISHED!".format(n_modes))
                break

        logger.info("  Train VGM with full data")
        model.fit(data)

        means = model.means_.reshape((1, n_modes))
        stds = np.sqrt(model.covariances_).reshape((1, n_modes))

        # Normalization
        normalized_values = ((data - means) / (self.std_span * stds))
        probs = model.predict_proba(data)

        # Clip the values
        normalized_values = np.clip(normalized_values, -.99, .99)

        return normalized_values, probs, model, n_modes

    def inverse_transform(self, data, info):
        """
        Reverse the clustering of values.

        Parameters
        ----------
            data: numpy.ndarray
                Transformed data to restore.
            info: dict
                Metadata.

        Returns
        -------
            numpy.ndarray
                Values in the original space.
        """

        gmm = info['transform']
        n_modes = info['n']

        normalized_values = data[:, :n_modes]
        probs = data[:, n_modes:]

        selected_component = select_values(probs, argmax=self.argmax)

        means = gmm.means_.reshape([-1])
        stds = np.sqrt(gmm.covariances_).reshape([-1])

        mean_t = means[selected_component]
        std_t = stds[selected_component]

        selected_normalized_value = normalized_values[np.arange(len(data)), selected_component]

        return selected_normalized_value * self.std_span * std_t + mean_t


class DATGANDataFlow(RNGDataFlow):
    """
    Subclass of tensorpack.RNGDataFlow prepared to work with numpy.ndarray.

    Same as in TGAN.

    Attributes
    ----------
        shuffle: bool
            Whether or not to shuffle the data.
        metadata: dict
            Metadata for the given `data`.
        num_features: int
            Number of features in given `data`.
        data: list
            Prepared data from `filename`.
    """

    def __init__(self, data, metadata, var_order, shuffle=True):
        """
        Initialize object.

        Parameters
        ----------
            data: str
                Path to the json file containing the metadata.
            metadata: dict
                Description of the inputs.
            var_order: list[str]
                Ordered list of the variables
            shuffle: bool, default True
                Whether or not to shuffle the data.

        Raises
        ------
            ValueError
                If any column_info['type'] is not supported

        """
        self.shuffle = shuffle
        if self.shuffle:
            self.reset_state()

        self.metadata = metadata
        self.num_features = self.metadata['num_features']

        self.data = []

        for col in var_order:
            column_info = self.metadata['details'][col]
            if column_info['type'] == 'continuous':
                col_data = data[col]
                n = column_info['n']
                value = col_data[:, :n]
                cluster = col_data[:, n:]

                self.data.append(value)
                self.data.append(cluster)

            elif column_info['type'] == 'category':
                col_data = np.asarray(data[col], dtype='int32')
                self.data.append(col_data)

            else:
                raise ValueError(
                    "column_info['type'] must be either 'category' or 'continuous'."
                    "Instead it was '{}'.".format(column_info['type'])
                )

        self.data = list(zip(*self.data))

    def size(self):
        """
        Return the number of rows in data.

        Returns
        -------
            int
                Number of rows in `data`.

        """
        return len(self.data)

    def get_data(self):
        """
        Yield the rows from `data`.

        Yields
        ------
            tuple
                Row of data.

        """
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            yield self.data[k]

    def __iter__(self):
        """Iterate over self.data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.size()


class RandomZData(DataFlow):
    """
    Random dataflow.

    Same as in TGAN.

    Arguments
    ---------
        shape: tuple
            Shape of the array to return on `get_data`

    """

    def __init__(self, shape):
        """Initialize object."""
        super(RandomZData, self).__init__()
        self.shape = shape

    def get_data(self):
        """Yield random normal vectors of shape `shape`."""
        while True:
            yield [np.random.normal(0, 1, size=self.shape)]

    def __iter__(self):
        """Return data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.shape[0]


def check_metadata(metadata):
    """
    Check that the given metadata has correct types for all its members.

    Same function as in TGAN.

    Parameters
    ----------
        metadata: dict
            Description of the inputs.

    Raises
    ------
        AssertionError
            If any of the details is not valid.

    """
    message = 'The given metadata contains unsupported types.'
    assert all([metadata['details'][col]['type'] in ['category', 'continuous']
                for col in metadata['details'].keys()]), message


def check_inputs(function):
    """
    Validate inputs for functions whose first argument is a numpy.ndarray with shape (n,1).

    Same function as in TGAN.

    Parameters
    ----------
        function: callable
            Method to validate.

    Returns
    -------
        callable
            Will check the inputs before calling `function`.

    Raises
    ------
        ValueError
            If first argument is not a valid`numpy.array of shape (n, 1).

    """
    def decorated(self, data, *args, **kwargs):
        if not (isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 1):
            raise ValueError('The argument `data` must be a numpy.ndarray with shape (n, 1).')

        return function(self, data, *args, **kwargs)

    decorated.__doc__ = function.__doc__
    return decorated


def select_values(probs, argmax):
    """
    Get a selected value in a discrete probability vector using either argmax or simulating

    Parameters
    ----------
    probs: list(float)
        Discrete probability vector
    argmax: bool
        If set to True, select a value in `probs` using argmax. Otherwise, uses simulation.

    Returns
    -------
        float:
            Single value selected from `probs`
    """
    if argmax:
        return np.argmax(probs, axis=1)
    else:
        probs = probs + 1e-6
        probs = np.divide(probs, np.sum(probs, axis=1).reshape((-1, 1)))
        c = probs.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        sel_comp = (u < c).argmax(axis=1)

        return sel_comp
