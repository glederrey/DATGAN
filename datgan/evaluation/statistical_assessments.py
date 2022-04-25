#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import combinations


def stats_assessment(original_data, synthetic_data, continuous_columns, aggregation_level, sep='::', ignore_cols=[]):
    """
    Compute all the stats on the different combinations based on the aggregation_level.

    Parameters
    ----------
    original_data: pandas.DataFrame
        Original dataset
    synthetic_data: pandas.DataFrame
        Synthetic dataset
    continuous_columns: list[str]
        List of continuous columns
    aggregation_level: int
        Aggregation level for the stats. Only accepts 1, 2, or 3.
    sep: str, default '::'
        String used to separated the columns while aggregating them.
    ignore_cols: list[str], default []
        List of columns to ignore while computing the stats.

    Returns
    -------
    stats: dict
        Dictionary containing all the stats for all the combinations
    """
    original_data = original_data.copy()
    synthetic_data = synthetic_data.copy()

    # First step is to discretize the continuous columns on both datasets
    bins = get_bins(original_data, continuous_columns)

    for c in continuous_columns:
        original_data[c] = pd.cut(original_data[c], bins=bins[c])
        synthetic_data[c] = pd.cut(synthetic_data[c], bins=bins[c])

    cols_to_treat = set(original_data.columns) - set(ignore_cols)

    # Get the combinations based on the aggregation level
    combs = get_combinations(cols_to_treat, aggregation_level, sep)

    stats = {}
    for c in combs:
        stats[c] = {}

        agg_vars = c.split('::')

        orig = original_data.copy()
        orig['count'] = 1
        orig = orig.groupby(agg_vars, observed=True).count()
        orig /= len(original_data)

        synth = synthetic_data.copy()
        synth['count'] = 1
        synth = synth.groupby(agg_vars, observed=True).count()
        synth /= len(synthetic_data)

        real_and_sampled = pd.merge(orig, synth, suffixes=['_orig', '_synth'], on=agg_vars, how='outer', indicator=True)
        real_and_sampled = real_and_sampled[['count_orig', 'count_synth']].fillna(0)

        sts = compute_stats(real_and_sampled['count_orig'], real_and_sampled['count_synth'])

        stats[c] = sts

    return stats


def get_combinations(columns, aggregation_level, sep='::'):
    """

    Parameters
    ----------
    columns: list[str]
        List of column names
    aggregation_level: int
        Aggregation level for the stats. Only accepts 1, 2, or 3
    sep: str, default '::'
        String used to separated the columns while aggregating them

    Raises
    ------
    ValueError:
        If a wrong aggregation level is given

    Returns
    -------
    combinations_list: list[str]
        The list of combinations of columns. Each element corresponds to one combinations.
    """

    if aggregation_level not in [1, 2, 3]:
        raise ValueError("The variable 'aggregation_level' can only take the values 1, 2, or 3.")

    combinations_list = []

    for k in combinations(columns, aggregation_level):
        str_ = ''
        for i in range(aggregation_level):
            str_ += k[i]
            if i < aggregation_level-1:
                str_ += sep

        combinations_list.append(str_)

    return combinations_list


def compute_stats(freq_list_orig, freq_list_synth):
    """
    Compute different statistics (MAE, RMSE, SMRSE, R^2, and Pearson's correlation) on two frequency lists.

    Parameters
    ----------
    freq_list_orig: numpy.ndarray
        Frequency list for the original data
    freq_list_synth: numpy.ndarray
        Frequency list for the synthetic data

    Returns
    -------
    stat: dict
        Dictionary of the stats between the two lists
    """

    freq_list_orig, freq_list_synth = np.array(freq_list_orig), np.array(freq_list_synth)
    corr_mat = np.corrcoef(freq_list_orig, freq_list_synth)
    corr = corr_mat[0, 1]
    if np.isnan(corr): corr = 0.0
    # MAE
    mae = np.absolute(freq_list_orig - freq_list_synth).mean()
    # RMSE
    rmse = np.linalg.norm(freq_list_orig - freq_list_synth) / np.sqrt(len(freq_list_orig))
    # SRMSE
    freq_list_orig_avg = freq_list_orig.mean()
    srmse = rmse / freq_list_orig_avg
    # r-square
    u = np.sum((freq_list_synth - freq_list_orig) ** 2)
    v = np.sum((freq_list_orig - freq_list_orig_avg) ** 2)
    r2 = 1.0 - u / v
    stat = {'mae': mae, 'rmse': rmse, 'r2': r2, 'srmse': srmse, 'corr': corr}

    return stat


def get_bins(data, continuous_cols):
    """
    Returns the bins for the continuous columns of the data.

    Parameters
    ----------
    data: pandas.DataFrame
        Original data
    continuous_cols: list[str]
        List of continuous columns

    Returns
    -------
    bins_cont: dict
        Dictionary containing the bins for transforming the continuous columns
    """

    bins_cont = {}

    for c in continuous_cols:
        bins_cont[c] = pd.cut(data[c], bins=10, retbins=True)[1]
        bins_cont[c][0] = -np.inf
        bins_cont[c][-1] = np.inf

    return bins_cont
