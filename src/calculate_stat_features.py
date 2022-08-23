import pandas as pd
import numpy as np

from sklearn import metrics


def calculate_statistical_features(grouped_data):
    # max
    stat_features = grouped_data.max()
    stat_features = stat_features.add_prefix('max_')

    # min
    min_ = grouped_data.min()
    min_ = min_.add_prefix('min_')
    stat_features = stat_features.join(min_)

    # auc
    auc = grouped_data.agg(auc_group)
    auc = auc.add_prefix('auc_')
    stat_features = stat_features.join(auc)

    # var
    var_ = grouped_data.var()
    var_ = var_.add_prefix('var_')
    stat_features = stat_features.join(var_)

    # sum_diffs
    sum_diffs = grouped_data.transform(sum_abs_diffs)
    sum_diffs = sum_diffs.add_prefix('sum_diffs_')
    stat_features = stat_features.join(sum_diffs)

    return stat_features


def auc_group(data=None, x=None, y=None):
    if x is None:
        x = data.index.astype('int64')
        y = data.values
    return metrics.auc(x, y)


def sum_abs_diffs(x):
    return np.sum(np.abs(np.diff(x)))
