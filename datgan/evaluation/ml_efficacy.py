#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from datgan.evaluation.LightGBM import LightGBMCV, emae, emse


def ml_assessment(df_orig, df_synth, continuous_columns, categorical_columns, params=None):
    """
    Train the LightGBMCV model on all the columns of the original and the synthetic datasets.

    Coded by Tim Hillel.

    Parameters
    ----------
    df_orig: pandas.DataFrame
        Original dataset
    df_synth: pandas.DataFrame
        Synthetic dataset
    continuous_columns: list[str]
        List of continuous variables
    categorical_columns: list[str]
        List of categorical variables
    params: dict
        Dictionary of parameters passed to the LightGBMCV model

    Returns
    -------
    dict:
        Dictionary with the variable names as keys and results as value

    """
    if not params:
        params = {'n_estimators': 5000}

    tmp = {}
    for k, ycol in enumerate(df_orig.columns):
        info = '    Column: {} ({}/{})'.format(ycol, k + 1, len(df_orig.columns))
        print(info, end="")
        sys.stdout.flush()
        Xcols = [c for c in df_orig.columns if c != ycol]

        y_synth = df_synth[ycol]
        X_synth = df_synth[Xcols]
        y_real = df_orig[ycol]
        X_real = df_orig[Xcols]

        observe_sets = {'original': (X_real, y_real)}
        ccols = [c for c in categorical_columns if c != ycol]

        if ycol in categorical_columns:
            lgbm_type = 'LGBMClassifier'
            kf = StratifiedKFold(shuffle=True, random_state=42)
            eval_metric = ['error']
        elif ycol in continuous_columns:
            lgbm_type = 'LGBMRegressor'
            kf = KFold(shuffle=True, random_state=42)
            eval_metric = ['l2', 'l1']
        cv = LightGBMCV(lgbm_type=lgbm_type,
                        splitter=kf,
                        eval_metric=eval_metric,
                        observe_sets=observe_sets,
                        separate_observation_split=True,
                        early_stopping_rounds=5,
                        return_cv_models=False,
                        refit_model=False,
                        verbose=True)
        cv.fit(X_synth, y_synth, categorical_feature=ccols, params=params)
        tmp[ycol] = cv.result_dict_

        print(' ' * len(info), end='\r')

        if k == len(df_orig.columns):
            print('', end='\r')

    return tmp

def transform_results(results, continuous_columns, categorical_columns):
    """
    Transform the results of the function `ml_assessment` in human-readable format.

    Coded by Tim Hillel.

    Parameters
    ----------
    results: dict
        Dictionary of results from the `ml_assessment` function. Keys corresponds to synthetic files testes, value to
        the dictionary returned by the `ml_assessment` function.
    continuous_columns: list[str]
        List of continuous variables
    categorical_columns: list[str]
        List of categorical variables

    Returns
    -------
        cont_sorted: list[tuple]
            Synthetic dataset sorted based on the results on continuous columns
        cat_sorted: list[tuple]
            Synthetic dataset sorted based on the results on categorical columns
    """

    ori_scores = {col: results['original'][col]['test_log_loss'] for col in categorical_columns}
    ori_scores.update({col: results['original'][col]['test_l2'] for col in continuous_columns})

    internal = {}
    external = {}
    external_normalised = {}
    cont_scores = {}
    cat_scores = {}

    for model in results:

        internal[model] = {}
        external[model] = {}
        external_normalised[model] = {}
        for col in categorical_columns:
            internal[model][col] = results[model][col]['test_log_loss']
            external[model][col] = results[model][col]['original_log_loss']
            external_normalised[model][col] = external[model][col] - ori_scores[col]

        for col in continuous_columns:
            internal[model][col] = results[model][col]['test_l2']
            external[model][col] = results[model][col]['original_l2']
            external_normalised[model][col] = external[model][col] - ori_scores[col]

        cont_scores[model] = sum([external[model][col] / ori_scores[col] for col in continuous_columns])
        cat_scores[model] = sum([external[model][col] - ori_scores[col] for col in categorical_columns])

    cat_sorted = sorted(cat_scores.items(), key=lambda item: item[1])
    cont_sorted = sorted(cont_scores.items(), key=lambda item: item[1])

    return cont_sorted, cat_sorted
