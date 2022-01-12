#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class for the Cross-Validation of LightGBM.

Coded by Tim Hillel
"""

import sys
import time
import warnings
import numpy as np
from typing import Callable
from collections.abc import Iterable
from lightgbm.callback import early_stopping
from sklearn.metrics import log_loss, mean_squared_error
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

class LightGBMCV:
    def __init__(self,
                 lgbm_type: str,
                 splitter: Callable,
                 eval_metric=None,
                 eval_sets: dict = None,
                 observe_sets: dict = None,
                 separate_observation_split: bool = True,
                 early_stopping_rounds: int = None,
                 return_cv_models: bool = False,
                 refit_model: bool = False,
                 verbose: bool = True):
        self.lgbm_type = lgbm_type
        self.splitter = splitter
        self.eval_metric = eval_metric
        self.eval_sets = eval_sets
        self.observe_sets = observe_sets
        self.separate_observation_split = separate_observation_split
        self.early_stopping_rounds = early_stopping_rounds
        self.return_cv_models = return_cv_models
        self.refit_model = refit_model
        self.verbose = verbose

    def fit(self, X, y, categorical_feature='auto', params=None):
        clf = False
        ordi = False
        reg = False
        metric = None
        alias = None
        metric_names = []
        metric_values = []
        self.classes_ = y.unique()
        if self.lgbm_type == 'LGBMClassifier':
            metric = 'binary_logloss'
            if len(self.classes_) > 2:
                metric = 'multi_logloss'
            alias = 'log_loss'
            clf = True
            self.estimator_ = LGBMClassifier
            y = y.astype(int)
        elif self.lgbm_type == 'LGBMRegressor':
            metric = 'l2'
            alias = 'l2'
            reg = True
            self.estimator_ = LGBMRegressor
        else:
            raise ValueError('lgbm_type must be LGBMClassifier, LGBMRegressor, or LGBMOrdinal')
        if not self.eval_metric:
            self.eval_metric = []
        elif isinstance(self.eval_metric, dict):
            self.eval_metric = [self.eval_metric]
        elif not isinstance(self.eval_metric, Iterable):
            self.eval_metric = [self.eval_metric]
        if metric not in self.eval_metric:
            self.eval_metric.insert(0, metric)
        for m in self.eval_metric:
            if isinstance(m, str):
                metric_names.append(m)
                metric_values.append(m)
            # elif isinstance(m, dict):
            #     for k, v in m.items():
            #         metric_names.append(k)
            #         if callable(v):
            #             hack = lambda y, x, name=k: (name, v(y, x), False)
            #             _v = lambda y, x: hack(y,x)
            #         else:
            #             _v = v
            #         metric_values.append(_v)
            # elif isinstance(m, Iterable):
            #     k=m[0]
            #     v=m[1]
            #     metric_names.append(k)
            #     if callable(v):
            #         hack = lambda y, x, name=k: (name, v(y, x), False)
            #         _v = lambda y, x: hack(y,x)
            #     else:
            #         _v = v
            #     metric_values.append(_v)
            else:
                raise ValueError(
                    'For now, eval_metric can only be a string or list of strings. Callables are yet to be implemented.')

        if self.lgbm_type == 'LGBMClassifier':
            metric_values = ['binary_error' if v == 'error' else v for v in metric_values]
            if len(y.unique()) > 2:
                metric_values = ['multi_error' if v == 'error' else v for v in metric_values]

        self.result_dict_ = {'train_scores': {m: [] for m in metric_names},
                             'test_scores': {m: [] for m in metric_names},
                             'best_iteration': [],
                             'fit_times': [],
                             'train_size': [],
                             'test_size': [],
                             'train_scores_mean': {},
                             'test_scores_mean': {}
                             }
        if params:
            self.result_dict_['specified_params'] = params
        if self.eval_sets:
            for k in self.eval_sets.keys():
                self.result_dict_[f'{k}_scores'] = {m: [] for m in metric_names}
                self.result_dict_[f'{k}_scores_mean'] = {}
        if self.observe_sets:
            for k in self.observe_sets.keys():
                self.result_dict_[f'{k}_{alias}_scores'] = []
        if self.return_cv_models:
            self.cv_models_ = []
        ifold = 0
        for train_i, test_i in self.splitter.split(X, y):
            if self.early_stopping_rounds:
                _callbacks = [
                    early_stopping(self.early_stopping_rounds,
                                   first_metric_only=True,
                                   verbose=False)]
            if self.verbose:
                str_ = ' -- Fold {}/{}'.format(ifold + 1, self.splitter.get_n_splits())

                if ifold > 0:
                    for i in range(len(str_)):
                        print('\b', end='')
                        sys.stdout.flush()

                print(str_, end="")
                sys.stdout.flush()

                if ifold + 1 == self.splitter.get_n_splits():

                    for i in range(len(str_)):
                        print('\b', end='')
                        sys.stdout.flush()

                    print('', end='\r')
                    sys.stdout.flush()

            if params:
                _est = self.estimator_(**params)
            else:
                _est = self.estimator_()
            X_train = X.iloc[train_i]
            y_train = y.iloc[train_i]
            X_test = X.iloc[test_i]
            y_test = y.iloc[test_i]
            if clf or ordi:
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)
                X_test = X_test[y_test.isin(y_train.unique())]
                y_test = y_test[y_test.isin(y_train.unique())]
            self.result_dict_['train_size'].append(train_i.shape[0])
            self.result_dict_['test_size'].append(test_i.shape[0])
            eval_set = [(X_train, y_train),
                        (X_test, y_test)]
            if self.eval_sets:
                for v in self.eval_sets.values():
                    eval_set.append((v[0].iloc[test_i], v[1].iloc[test_i]))
            ts = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                _est.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    eval_metric=metric_values,
                    categorical_feature=categorical_feature,
                    callbacks=_callbacks
                )
            self.result_dict_['fit_times'].append(time.time() - ts)
            if _est.best_iteration_:
                self.result_dict_['best_iteration'].append(_est.best_iteration_)
            train_name = list(_est.best_score_.keys())[0]
            test_name = list(_est.best_score_.keys())[1]
            for i, m in enumerate(metric_names):
                _k = m

                if m == 'error':
                    _k = metric_values[i]

                # Very specific bug between binary and multi logloss
                if _k not in _est.best_score_[train_name].keys():
                    _k = 'binary_logloss'

                self.result_dict_['train_scores'][m].append(_est.best_score_[train_name][_k])
                self.result_dict_['test_scores'][m].append(_est.best_score_[test_name][_k])
                if self.eval_sets:
                    for j, k in enumerate(self.eval_sets.keys()):
                        res_name = list(_est.best_score_.keys())[j + 2]
                        self.result_dict_[f'{k}_scores'][m].append(_est.best_score_[res_name][m])
            if self.observe_sets:
                for k, v in self.observe_sets.items():
                    X_obs = v[0]
                    y_obs = v[1]
                    if self.separate_observation_split:
                        test_i = list(self.splitter.split(X_obs, y_obs))[i][1]
                    X_test = X_obs.iloc[test_i]
                    y_test = y_obs.iloc[test_i]
                    if clf or ordi:
                        try:
                            s = log_loss(_est._le.transform(y_test), _est.predict_proba(X_test),
                                         labels=_est._le.classes_)
                            self.result_dict_[f'{k}_{alias}_scores'].append(s)
                        except:
                            self.result_dict_[f'{k}_{alias}_scores'].append(9999)
                    elif reg:
                        s = mean_squared_error(y_test, _est.predict(X_test))
                        self.result_dict_[f'{k}_{alias}_scores'].append(s)
            if self.return_cv_models:
                self.cv_models_.append(_est)
            ifold += 1
        for m in metric_names:
            self.result_dict_['train_scores_mean'][m] = sum(
                [self.result_dict_['train_scores'][m][i] * self.result_dict_['train_size'][i] for i in
                 range(len(self.result_dict_['train_scores'][m]))]) / sum(self.result_dict_['train_size'])
            self.result_dict_['test_scores_mean'][m] = sum(
                [self.result_dict_['test_scores'][m][i] * self.result_dict_['test_size'][i] for i in
                 range(len(self.result_dict_['test_scores'][m]))]) / sum(self.result_dict_['test_size'])
            if self.eval_sets:
                for j, k in enumerate(self.eval_sets.keys()):
                    res_name = list(_est.best_score_.keys())[j + 2]
                    self.result_dict_[f'{k}_scores_mean'][m] = sum(
                        [self.result_dict_[f'{k}_scores'][m][i] * self.result_dict_['test_size'][i] for i in
                         range(len(self.result_dict_[f'{k}_scores'][m]))]) / sum(self.result_dict_['test_size'])
        self.result_dict_[f'train_{alias}'] = self.result_dict_['train_scores_mean'][metric]
        self.result_dict_[f'test_{alias}'] = self.result_dict_['test_scores_mean'][metric]
        self.train_score_ = self.result_dict_[f'train_{alias}']
        self.test_score_ = self.result_dict_[f'test_{alias}']
        if self.eval_sets:
            for k in self.eval_sets.keys():
                self.result_dict_[f'{k}_{alias}'] = self.result_dict_[f'{k}_scores_mean'][metric]
        if self.observe_sets:
            for k in self.observe_sets.keys():
                self.result_dict_[f'{k}_{alias}'] = sum(
                    [self.result_dict_[f'{k}_{alias}_scores'][i] * self.result_dict_['test_size'][i] for i in
                     range(len(self.result_dict_['test_scores'][m]))]) / sum(self.result_dict_['test_size'])
        self.result_dict_['params'] = _est.get_params()
        self.result_dict_['fit_times_mean'] = sum(self.result_dict_['fit_times']) / len(self.result_dict_['fit_times'])
        if 'best_iteration' in self.result_dict_.keys():
            if self.result_dict_['best_iteration']:
                self.best_iteration_max_ = max(self.result_dict_['best_iteration'])
        if self.refit_model:
            print('refitting final model')
            _est = self.estimator_()
            if hasattr(self, self.best_iteration_max_):
                if not params:
                    params = {}
                params['n_estimators'] = self.best_iteration_max_
            if params:
                _est.set_params(**params)
            _est.fit(X, y)
            self.refitted_model_ = _est
        return self


def emae(y_true, preds):
    if preds.shape[0] == len(y_true):
        try:
            if preds.shape[1]<=1:
                raise ValueError('''preds must have width of n_classes''')
        except:
            raise ValueError(f'''preds must have shape (len(y_true), J) or (len(y_true)*J,). 
            It appears preds has shape len(y_true,), i.e. {preds.shape}''')
        y_prob=preds
    elif preds.shape[0]%len(y_true)==0:
        y_prob = preds.reshape(int(preds.shape[0]/len(y_true)),-1).T
    else:
        raise ValueError(f'preds is weird shape of {preds.shape}')
    diffs = np.abs(np.tile(np.arange(y_prob.shape[1]), (y_prob.shape[0], 1)) - np.tile(y_true, (y_prob.shape[1],1)).T)
    return ((diffs*y_prob).sum(axis=1)).mean()


def emse(y_true, preds):
    if preds.shape[0] == len(y_true):
        try:
            if preds.shape[1]<=1:
                raise ValueError('''preds must have width of n_classes''')
        except:
            raise ValueError(f'''preds must have shape (len(y_true), J) or (len(y_true)*J,). 
            It appears preds has shape len(y_true,), i.e. {preds.shape}''')
        y_prob=preds
    elif preds.shape[0]%len(y_true)==0:
        y_prob = preds.reshape(int(preds.shape[0]/len(y_true)),-1).T
    else:
        raise ValueError(f'preds is weird shape of {preds.shape}')
    diffs = (np.tile(np.arange(y_prob.shape[1]), (y_prob.shape[0], 1)) - np.tile(y_true, (y_prob.shape[1],1)).T)**2
    return ((diffs*y_prob).sum(axis=1)).mean()