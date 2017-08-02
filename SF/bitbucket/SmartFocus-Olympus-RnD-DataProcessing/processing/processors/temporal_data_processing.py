from __future__ import division, print_function
import numpy as np
import pandas
import itertools
import processing.constants as constants
import processing.utils.distances as dist
import processing.utils.filters as filters
from processing.utils.analytics import mad, sort_dataframe
from processing.utils.jointpdfsampling import fill_missing_values_rolling
import ipdb
from . import utils
import joblib
import time

import scipy.stats as stats


def timerange(x):
    return (x.max() - x.min()) * 1E-9


def count_observations(x):
    return x.size


def normalize_time(x):
    if 'timestamp' in x.columns:
        return x['timestamp'] - x['timestamp'].min()
    else:
        x.sort_index(inplace=True)
        x.index = pandas.Index(x.index.values - x.index.min(), name='timestamp')
        return x


def create_features_from_raw_data(raw_data_frame, **kwargs):
    """create features from raw data

    Args:
        raw_data_frame: the dataframe with the raw data
        **kwargs:   workers = kwargs.pop("workers", 8)
                    missing_value = kwargs.pop("missing_value", constants.MISSING_BEACON_DATA_VALUE)
                    window_size = kwargs.pop("window_size", 10)
                    window_step = kwargs.pop("window_step", 2)
                    features = kwargs.pop("features", ["median", "max"])
                    drop_low_observations = kwargs.pop("drop_low_observations", False)
                    channel = kwargs.pop("channel", '0')

    Returns:

    """
    workers = kwargs.pop("workers", 8)
    configuration = kwargs.pop("configuration")

    if configuration.channel != '0':
        raw_data_frame = raw_data_frame.loc[raw_data_frame["beacon"].str.contains('^' + configuration.channel)]

    if configuration.filter_beacons:
        raw_data_frame = raw_data_frame.loc[~raw_data_frame['beacon'].isin(configuration.filter_beacons)]
        print("filtered beacons validation data shape:{}".format(raw_data_frame.shape))

    splits = itertools.product(raw_data_frame['node'].unique(), ['N', 'E', 'S', 'W'])
    start_time = time.time()
    with joblib.Parallel(n_jobs=workers, backend='multiprocessing', verbose=0,
                         pre_dispatch='2.0*n_jobs') as parallel:
        results = parallel(
            joblib.delayed(wrangle_data_temporal, check_pickle=False)(
                raw_data_frame.loc[np.logical_and(raw_data_frame.node == node,
                                                  raw_data_frame.direction == direction)],
                window_size=configuration.window_size,
                window_step=configuration.window_step,
                features=configuration.features,
                feature_type=configuration.validation_type,
            )
            for node, direction in splits)
    end_time = time.time()
    print('time taken: {}'.format(end_time - start_time))

    rolled_data = pandas.concat([result for result in results if result.size > 0])
    if 0 in rolled_data.columns:
        rolled_data.drop(0, axis=1, inplace=True)

    transformed_data = wrangle_temporal_data(
        rolled_data,
        missing_value=configuration.missing_value,
        drop_low_observations=configuration.drop_low_observations,
        fill_low_obs_threshold=configuration.fill_low_obs_threshold,
        observed_value=configuration.observed_value,
        window_size=configuration.window_size,
        frequency_threshold=configuration.frequency_threshold,
    )
    transformed_data.columns = pandas.Index(['win_' + str(i) for i in transformed_data.columns], name='window')
    return transformed_data


def roll_group(data_fingerprint, **kwargs):
    """

    Args:
        data_fingerprint: a pandas.Series() with the data for the beacon
        **kwargs: window_size: the size of the window
                  window_step: the step of the rolling window
                  features: the features that will be used

    Returns:

    """

    window_size_seconds = kwargs.pop('window_size')
    window_step = kwargs.pop('window_step')
    threshold = kwargs.pop("threshold", 20)
    if not window_size_seconds or not window_step:
        raise Exception("no window size or no window_step")
    features = kwargs.pop('features', ['median', 'max'])
    data_fingerprint.sort_index(inplace=True)
    window_size = pandas.to_timedelta(window_size_seconds, 'S')
    window_step = pandas.to_timedelta(window_step, 'S')
    current_time = data_fingerprint.index[0]
    end = data_fingerprint.index[-1]
    assert (end - current_time).total_seconds() < 60 * 3
    feature_dic = {
        "mean": lambda x: np.nan if x.size == 0 else np.nanmean(x.values),
        "median": lambda x: np.nan if x.size == 0 else np.nanmedian(x.values),
        "std": lambda x: np.nan if x.size == 0 else np.nanstd(x.values),
        "max": lambda x: np.nan if x.size == 0 else np.nanmax(x.values),
        "min": lambda x: np.nan if x.size == 0 else np.nanmin(x.values),
        "obs": lambda x: 0 if x.size == 0 else np.sum(~np.isnan(x.values)),
        "frequency": lambda x: 0.0 if x.size == 0 else np.sum(~np.isnan(x.values)) / float(window_size_seconds),
        "quant3": lambda x: np.nan if x.size == 0 else x.quantile(0.75),
        "observed": lambda x: -2 if x.size == 0 else 2
    }
    if 'obs' not in features:
        features.append('obs')
    result = []
    while (current_time <= end):
        row_results = [feature_dic[feature](data_fingerprint.loc[current_time:current_time + window_size])
                       for feature in features]
        row = pandas.Series(row_results,
                            index=features
                            )
        result.append(row)

        current_time += window_step

    feature_data = pandas.concat(result, axis=1).transpose()
    feature_data.index.rename('window', inplace=True)
    return feature_data


def expand_group(data_fingerprint, **kwargs):
    """

    Args:
        data_fingerprint: a pandas.Series() with the data for the beacon
        **kwargs: window_size: the size of the window
                  window_step: the step of the rolling window
                  features: the features that will be used

    Returns:

    """

    window_size_seconds = kwargs.pop('window_size')
    window_step = kwargs.pop('window_step')
    threshold = kwargs.pop("threshold", 20)
    if not window_size_seconds or not window_step:
        raise Exception

    features = kwargs.pop('features', ['median', 'max'])
    data_fingerprint.sort_index(inplace=True)
    # ipdb.set_trace()
    window_size = pandas.to_timedelta(window_size_seconds, 'S')
    window_step = pandas.to_timedelta(window_step, 'S')
    start_time = data_fingerprint.index[0]
    current_time = start_time + window_size
    end = data_fingerprint.index[-1] + pandas.to_timedelta(1, 'S') + window_size
    assert (end - current_time).total_seconds() > 0, "total seconds are: {}".format(
        (end - current_time).total_seconds())
    feature_dic = {
        "mean": lambda x: np.nan if x.size == 0 else np.nanmean(x.values),
        "median": lambda x: np.nan if x.size == 0 else np.nanmedian(x.values),
        "std": lambda x: np.nan if x.size == 0 else np.nanstd(x.values),
        "max": lambda x: np.nan if x.size == 0 else np.nanmax(x.values),
        "min": lambda x: np.nan if x.size == 0 else np.nanmin(x.values),
        "obs": lambda x: 0 if x.size == 0 else np.sum(~np.isnan(x.values)),
        "frequency": lambda x: 0.0 if x.size == 0 else np.sum(~np.isnan(x.values)) / (
        x.index.max() - x.index.min()).total_seconds(),
        "quant3": lambda x: np.nan if x.size == 0 else x.quantile(0.75, interpolation='midpoint'),
        "observed": lambda x: -2 if x.size == 0 else 2
    }
    if 'obs' not in features:
        features.append('obs')
    if 'frequency' not in features:
        features.append('frequency')
    result = []
    # if data_fingerprint.shape[0]< 5: ipdb.set_trace()
    while (current_time <= end):
        row_results = [feature_dic[feature](data_fingerprint.loc[start_time:current_time])
                       for feature in features]
        row = pandas.Series(row_results,
                            index=features
                            )
        result.append(row)

        current_time += window_step

    # ipdb.set_trace()
    feature_data = pandas.concat(result, axis=1).transpose()
    feature_data.index.rename('window', inplace=True)
    return feature_data


def wrangle_temporal_data(aggregated_data, **kwargs):
    """

    Args:
        aggregated_data: grouped data to be stackec and moved
        **kwargs:  missing_value: the dbm value to add to missing values
                   drop_low_observations: if true observations with low observations will be modified

    Returns:

    """
    missing_value = kwargs.pop("missing_value", constants.MISSING_BEACON_DATA_VALUE)
    observed_value = kwargs.pop("observed_value", constants.MISSING_BEACON_DATA_VALUE)
    drop_low_observations = kwargs.pop("drop_low_observations", False)
    fill_low_obs_threshold = kwargs.pop("fill_low_obs_threshold", 10)
    validation_type = kwargs.pop("validation_type", 'expand')
    frequency_threshold = kwargs.pop("frequency_threshold", 2.0)
    window_size = kwargs.pop("window_size", 15)
    if 'frequency' not in aggregated_data:
        aggregated_data['frequency'] = aggregated_data['obs'] / (1.0 * window_size)

    try:
        unstacked_data = aggregated_data.unstack('window')
    except:
        raise Exception("data does not have window")
        # data_results = []
        # for set in unstacked_data.index.get_level_values('set').unique():
        #     data_results.append(aggregated_data.xs(set,level='set').unstack('window'))

    unstacked_data.dropna(axis=1, thresh=unstacked_data.shape[0] * 0.6, inplace=True)  # really small window

    def null_to_max(data, column_name):
        max_obs = data[column_name].max(axis=1)
        for column in data[column_name]:
            data.loc[unstacked_data.loc[:, (column_name, column)].isnull(), (column_name, column)] = max_obs.loc[
                data.loc[:, (column_name, column)].isnull()]
        return data

    if validation_type == 'expand':
        unstacked_data = null_to_max(unstacked_data, 'obs')
    else:
        unstacked_data['obs'] = unstacked_data['obs'].fillna(0)

    if 'std' in unstacked_data.columns:
        unstacked_data['std'] = unstacked_data['std'].fillna(0.1)
    if 'mad' in unstacked_data.columns:
        unstacked_data.loc['mad'] = unstacked_data.loc['mad'].fillna(1)
    if 'frequency' in unstacked_data.columns:
        if validation_type == 'expand':
            unstacked_data = null_to_max(unstacked_data, 'frequency')
        else:
            unstacked_data['frequency'] = unstacked_data['frequency'].fillna(0)

    if 'observed' in unstacked_data.columns:
        unstacked_data['observed'] = unstacked_data['observed'].fillna(-2)

    unstacked_data.fillna(missing_value, inplace=True)  # same value for min, max, median and mean

    if drop_low_observations:
        unstacked_data = utils.fill_low_observations(
            unstacked_data,
            fill_low_obs_threshold=fill_low_obs_threshold,
            missing_value=missing_value,
            observed_value=observed_value,
            frequency_threshold=frequency_threshold
        )

    unstacked_data.drop(['obs', 'frequency'], inplace=True, axis=1)  # drop observations
    unstacked_data.columns.set_names(['stat', 'window'], inplace=True)
    unstacked_data = unstacked_data.stack('stat')
    # FrozenList([u'node', u'direction', u'beacon', u'set', u'stat']) ->
    # FrozenList([u'set', u'node', u'direction', u'stat', u'beacon'])
    result = unstacked_data.reorder_levels(['set', 'node', 'direction', 'stat', 'beacon'],
                                           axis=0).sort_index(ascending=True)

    assert not np.any(result.isnull()), "no nulls in results"
    return result


def wrangle_data_temporal(raw_data_frame, **kwargs):
    """
    rolls the data in windows and gets you the statistics of each window


    Args:
        raw_data_frame:
        **kwargs: window_size the size of each window in seconds
                  window_step the step for each roll
                  features the features to use
                  drop_low_observations drop the low observations

    Returns:
        a dataframe with the windows and the features
    """
    window_size = kwargs.pop("window_size")
    window_step = kwargs.pop("window_step")
    features = kwargs.pop("features")
    feature_type = kwargs.pop("feature_type", 'expand')

    feature_functions = {"expand": expand_group, "roll": roll_group}

    if 'timestamp' in raw_data_frame.columns:
        raw_data_frame.loc[:, 'timestamp'] = pandas.to_datetime(raw_data_frame.loc[:, 'timestamp'])
        raw_data_frame.set_index('timestamp', inplace=True)

    groups = raw_data_frame.groupby(['node', 'direction', 'beacon', 'set'])
    aggregated = groups['rssi'].apply(lambda x: feature_functions[feature_type](x, window_size=int(window_size),
                                                                                window_step=int(window_step),
                                                                                features=features,
                                                                                ))
    return aggregated


def transform_temporal_data_to_features(rolled_data, direction, norm, pcaComponentsCount=None,
                                        missing_value=constants.MISSING_BEACON_DATA_VALUE
                                        ):
    test_data = []
    full_beacons = norm[0].index.get_level_values('beacon').unique().tolist()
    directed_rolled_data = rolled_data.xs(direction, level='direction')
    true_coordinates_list = directed_rolled_data.index.get_level_values(level='node').unique()

    for true_coordinates in true_coordinates_list:
        result = []

        position_data = directed_rolled_data.xs(true_coordinates, level='node')

        for fingerprint_index in position_data.index.get_level_values('set').unique():

            current_data = position_data.xs(fingerprint_index)

            windowed_data_beacons = set(current_data.index.get_level_values(level='beacon').tolist())

            windowed_data_full = fill_missing_values_rolling(current_data, all_beacons=full_beacons)
            for column in windowed_data_full.columns:
                column_data = windowed_data_full[column]
                if (column_data <= -100.0).mean() <= 0.97:  # not np.all(column_data == missing_value):
                    column_data = sort_dataframe((column_data - norm[0]) / norm[1], axis=0)
                    if pcaComponentsCount is not None:
                        column_data = np.dot(pcaComponentsCount, column_data.values)
                    else:
                        column_data = column_data.values
                    result.append(column_data)

        data = np.asarray(result)

        if data.size > 0:
            new_index = ['win_' + str(i) for i in range(data.shape[0])]
            column_index = windowed_data_full.index if pcaComponentsCount is None else pcaComponentsCount.index
            data = pandas.DataFrame(data, columns=column_index,
                                    index=new_index)
            data['node'] = true_coordinates
            test_data.append(data)

    result = pandas.concat(test_data, axis=0)
    result['direction'] = direction
    return result.set_index(['direction', 'node'])
