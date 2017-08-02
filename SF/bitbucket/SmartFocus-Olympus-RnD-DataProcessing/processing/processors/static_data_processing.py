from __future__ import print_function
import numpy as np
import scipy.stats as stats
import pandas
from sklearn.decomposition import PCA, RandomizedPCA
import processing.constants as constants
from .temporal_data_processing import timerange, count_observations
from processing.utils.analytics import mad, sort_dataframe
import processing.utils.filters as filters
from . import utils
from six import iteritems
import copy
import ipdb

def create_statistics_from_raw_data(raw_ble_data_df, **kwargs):
    """
    Args:
        raw_ble_data_df: dataframe with raw_Data
        **kwargs:  filter_data if true filter data
                    log_data if true use negative log of rssi

    Returns:
        dataframe with the statistics of the raw_fingerprints
    """

    if kwargs.pop("log_data"):
        raw_ble_data_df['rssi'] = np.log(-raw_ble_data_df['rssi'])

    grouped = raw_ble_data_df.groupby(['node', 'beacon', 'direction', 'set'])
    current_data_frequency = grouped['timestamp'].agg({"timerange": timerange, "obs": count_observations})
    area_grouped = raw_ble_data_df.groupby(['node', 'direction', 'set'])
    real_time_range = area_grouped.agg(timerange)['timestamp'].median()
    current_data_statistics = grouped['rssi'].agg({"min": np.min, "max": np.max,
                                                   "median": np.median, "mad": mad,
                                                   "mean": np.mean, "std": np.std,
                                                   "skew": stats.skew,
                                                   "kurtosis": stats.kurtosis,
                                                   'quant3': lambda x: x.quantile(0.75, interpolation='midpoint')
                                                   })

    current_data_frequency['frequency'] = current_data_frequency['obs'] / real_time_range
    current_data_statistics["observed"] = 2.0
    current_data_frequency['time_visibility'] = current_data_frequency['timerange'] / real_time_range
    current_data_statistics = pandas.concat([current_data_statistics, current_data_frequency], axis=1)
    return current_data_statistics


def create_features_from_statistics(statistics_df, **kwargs):
    """

    Args:
        statistics_df: a dataframe with the statistics of the fingerprints
        **kwargs:  missing_value the missing value to use
                features: a list with the features to get
                filter_beacons: a list of beacons to filter out
                channel: a channel to use

    Returns:

    """
    configuration = kwargs.pop("configuration", None)
    if configuration is not None:
        drop_low_observations = configuration.drop_low_observations
        fill_low_obs_threshold = configuration.fill_low_obs_threshold
        missing_value = configuration.missing_value
        observed_value = configuration.observed_value
        features = configuration.features
        filter_beacons = configuration.filter_beacons
        channel = configuration.channel
    else:
        drop_low_observations = kwargs.pop("drop_low_observations")
        fill_low_obs_threshold = kwargs.pop("fill_low_obs_threshold")
        missing_value = kwargs.pop("missing_value")
        observed_value = kwargs.pop("observed_value")
        features = kwargs.pop("features")
        filter_beacons = kwargs.pop("filter_beacons")
        channel = kwargs.pop("channel")

    if filter_beacons:
        statistics_df = statistics_df.loc[~statistics_df.index.get_level_values("beacon").isin(filter_beacons)]

        print("filtered beacons data shape:{}".format(statistics_df.shape))

    if channel != '0':
        statistics_df = statistics_df.loc[statistics_df.index.get_level_values("beacon").str.contains('^' + channel)]
        assert statistics_df.size>0, "no channel {} in data".format(channel)
    if drop_low_observations:
        statistics_df = utils.fill_low_observations(
            statistics_df,
            fill_low_obs_threshold=fill_low_obs_threshold,
            missing_value=missing_value,
            observed_value=observed_value,
            frequency_threshold=configuration.frequency_threshold
        )

    current_data_features = transform_features(
        statistics_df,
        missing_value=missing_value,
        features=features,
    )
    return current_data_features


def partition_function(group, split=0.75, less=True):
    if split <= 1:
        group_range = (group['timestamp'].max() - group['timestamp'].min()) * split + group['timestamp'].min()

    else:
        group_range = group['timestamp'].max() - split

    if less:
        partition = group[group['timestamp'] <= group_range]
    else:
        partition = group[group['timestamp'] > group_range]

    median_value = partition['rssi'].median()
    mad_value = mad(partition['rssi'])
    max_value = partition['rssi'].max()
    return pandas.Series([mad_value, max_value, median_value], index=['mad', 'max', 'median'])


def partition_groups(grouped, split_point=4E9):
    if split_point is 1:
        first_partition = grouped['rssi'].agg({"median": np.median, "mad": mad, "max": np.max})
        second_partition = pandas.DataFrame(index=first_partition.index, columns=first_partition.columns)
    else:
        first_partition = grouped.apply(lambda g: partition_function(g, split_point, less=True))
        second_partition = grouped.apply(lambda g: partition_function(g, split_point, less=False))

    return first_partition, second_partition


def transform_features(data_features,
                       missing_value=constants.MISSING_BEACON_DATA_VALUE,
                       features=None,
                       ):
    """Transforms data features dataframe by pivoting beacon from index to column
       And filling up data for missing beacons for each node

       Args:
           data_features the data features to be transformed

    """
    features = ["median", "quant3"] if features is None else features

    data_features = data_features.unstack('beacon')  # move beacons from index to column

    current_data_features = data_features.loc[:, features]

    current_data_features = current_data_features.dropna(
        how='all')  # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
    current_data_features.index.set_names(['node', 'direction', 'set'], inplace=True)
    current_data_features.columns.set_names(['stat', 'beacon'], inplace=True)

    if current_data_features.size == 0:
        return current_data_features

    if 'mad' in features:
        current_data_features.loc(axis=1)['mad', :] = current_data_features.loc(axis=1)['mad', :].fillna(1)
        current_data_features.replace({'mad': {0: 1}}, inplace=True)
    if 'frequency' in features:
        current_data_features.loc(axis=1)['frequency', :] = current_data_features.loc(axis=1)['frequency', :].fillna(0)
    if 'observed' in features:
        current_data_features.loc(axis=1)['observed', :] = current_data_features.loc(axis=1)['observed', :].fillna(-2.0)

    current_data_features.fillna(missing_value, inplace=True)

    assert not current_data_features.isnull().any().any(), "unstacked data statistics has null values"

    return current_data_features


def group_data_set_by_direction(data_set, features=None,
                                missing_value=constants.MISSING_BEACON_DATA_VALUE):
    """Groups a dataset, Removes non used features and groups the data_set by directions

    """
    features = ['max', 'median'] if features is None else features
    if data_set is None:
        return data_set, {}
    if 'mad' in features:
        data_set.loc(axis=1)['mad', :] = data_set.loc(axis=1)['mad', :].fillna(1)
        data_set.replace({'mad': {0: 1}}, inplace=True)
    if 'frequency' in features:
        data_set.loc(axis=1)['frequency', :] = data_set.loc(axis=1)['frequency', :].fillna(0)
    if "observed" in features:
        data_set.loc(axis=1)['observed', :] = data_set.loc(axis=1)['observed', :].fillna(-2.0)

    data_set.fillna(missing_value, inplace=True)

    data_set = data_set.swaplevel('direction', 'node').sort_index()
    data_set = data_set.reindex_axis(sorted(data_set.columns), axis=1)
    directed_data_set = {a[0]: {'data': sort_dataframe(a[1], axis=1)} for a in data_set.groupby(level="direction")}
    return data_set, directed_data_set


def group_data_features(data_features, validation_data, features=None,
                        missing_value=constants.MISSING_BEACON_DATA_VALUE
                        ):
    """Groups the data by direction

        Args:
            data_features (list): a list of all  calculated data features
            data_statistics (list): a list of all the calculated statistics
            raw_data (list): a list of all the raw_data
            validation_data (list): a list of all the validation data features
            use_mad (bool): use mad as a feature
            use_median (bool): use median as a feature
            use_max (bool): use max as a feature
            full (bool): if true return all the concatenated dataframes,
             otherwise return only the feature data_setand the unique beacons
        Returns:
            if full is true
            tuple(pandas.dataframe, dict, pandas.series): a tuple containaing the pandas.dataframe with the features,
                a dictionary with the dataset split by direction and a pandas.series with the unique beacons
            if full is false
             tuple(pandas.dataframe, dict, pandas.dataframe, pandas.dataframe, pandas.dataframe): a tuple containaing
              the pandas.dataframe with the features, a dictionary with the dataset split by direction and
               pandas.dataframes with the raw_data, the statistics and the frequency analysis

    """

    features = ["max", "median"] if features is None else features

    data_set, directed_data_set = group_data_set_by_direction(data_features, features=features,
                                                              missing_value=missing_value
                                                              )

    validation_set, directed_validation_set = group_data_set_by_direction(validation_data, features=features,
                                                                          missing_value=missing_value
                                                                          )

    for direction in directed_data_set.keys():
        if directed_validation_set.get(direction):
            directed_data_set[direction]['validation'] = directed_validation_set[direction]['data']
        else:
            directed_data_set[direction]['validation'] = None

    # data_analysis = pandas.concat(data_statistics)

    # raw_data_df = pandas.concat(raw_data, axis=0)
    # raw_data_df.columns = [u'timestamp', u'node', u'beacon', u'rssi', u'direction']
    return data_set, directed_data_set


def data_set_to_joint_distribution(directed_data_set):
    """Creates a dictionary with the joint distribution parameters for the directed data set

    data_set_to_joint_distribution finds the mean and covariance matrix for each direction and saves them to a dictionary

    Args:
        directed_data_set (dict): A dictionary with keys the directions ['N', 'E','S','W'], and values the data set
         values after preprocesing

    Returns:
        dict: A dictionary with a keys the directions and values a list containing
                the mean (pandas.Series) and covariance (pandas.DataFrame)



    """
    joint_distribution = {}
    for direction in constants.DIRECTIONS:
        joint_distribution[direction] = {}
        for key, value in iteritems(directed_data_set[direction]):
            if key == 'data':
                if 'obs' in value.columns:
                    value.drop("obs", axis=1,
                               inplace=True)  # PerformanceWarning: dropping on a non-lexsorted multi-index without a level parameter may impact performance.
                value.sort_index(inplace=True)
                joint_distribution[direction]['joint'] = [value.mean(), value.cov()]
                joint_distribution[direction]['data'] = directed_data_set[direction]['data']
                joint_distribution[direction]['fingerprints'] = copy.deepcopy(directed_data_set[direction]['data'])
            else:
                joint_distribution[direction][key] = directed_data_set[direction][key]
    return joint_distribution


def scale_data(directed_data_set):
    """Uses scales the directed_data_set so that each point has zero mean and one std

        Args:
         directed_data_set (dict): A dictionary with keys the directions and inside a dictionary
            with keys: 'data' for the data, 'joint' for the joint pdf

        Returns:
            dict: A dictionary where the 'data' value for each direction has been replaced with the normalized dataframe,
                an added key  'norm' has been added containing  a tuple with the mean and the std


    """

    data_set_dir_norm = {}
    for direction in constants.DIRECTIONS:
        data_set_dir_norm[direction] = {}
        for key in directed_data_set[direction].keys():
            if key == 'data':
                dir_data_mean = directed_data_set[direction]['data'].mean()

                dir_data_std = directed_data_set[direction]['data'].std().replace(0, 1)

                data_set_dir_norm[direction]['data'] = \
                    (directed_data_set[direction]['data'] - dir_data_mean) / dir_data_std
                data_set_dir_norm[direction]['norm'] = (
                    dir_data_mean, dir_data_std)
                if directed_data_set[direction].get('validation', None) is not None:
                    data_set_dir_norm[direction]['validation'] = \
                        (directed_data_set[direction]['validation'] - dir_data_mean) / dir_data_std
                else:
                    data_set_dir_norm[direction]['validation'] = None
            elif key == 'validation' and data_set_dir_norm[direction].get('validation') is None:
                data_set_dir_norm[direction]['validation'] = None
            else:
                data_set_dir_norm[direction][key] = directed_data_set[direction][key]

    return data_set_dir_norm


def reduce_dimensionality(directed_data_set, pcaComponentsCount, pca_type='PCA'):
    """Uses PCA or randomizedPCA to reduce dimensions of directed_data_set

        Args:
         directed_data_set (dict): A dictionary with keys the directions and inside a dictionary
            with keys: 'data' for the data, 'norm', for the normalization and 'joint' for the joint pdf
         pcaComponentsCount (int or float): If int the number of pcaComponentsCount to keep or if float the percentage of variance to keep

        Returns:
            dict: A dictionary where the 'data' value for each direction has been replaced with the pca
             features dataframe,an added key 'pcaComponentsCount' has been addedi with the pca pcaComponentsCount


    """
    pcaComponentsCount = int(pcaComponentsCount)
    if pcaComponentsCount <= 1:
        return directed_data_set
    assert pcaComponentsCount > 0, 'pcaComponets is {}'.format(pcaComponentsCount)
    data_set_dir_pca = {}
    for direction in constants.DIRECTIONS:
        data_set_dir_pca[direction] = {}
        for key in directed_data_set[direction].keys():
            if key == 'data':
                if pca_type == 'PCA':
                    dim_reduction_estimator = PCA(n_components=pcaComponentsCount)

                else:
                    assert type(pcaComponentsCount) is int, 'n_pcaComponentsCount is not int'
                    dim_reduction_estimator = RandomizedPCA(n_components=pcaComponentsCount)

                original_columns = directed_data_set[direction]['data'].columns

                data_set_dir_pca[direction]['data'] = pandas.DataFrame(
                    dim_reduction_estimator.fit_transform(directed_data_set[direction]['data'].values),
                    index=directed_data_set[direction]['data'].index
                )
                print("pca variance: {} / {} \n".format(dim_reduction_estimator.explained_variance_ratio_,
                                                        np.sum(dim_reduction_estimator.explained_variance_ratio_)
                                                        ))
                data_set_dir_pca[direction]['data'].columns = pandas.Index(
                    ['pca_' + str(i) for i in data_set_dir_pca[direction]['data'].columns]
                )
                print("estimator data size : {}".format(data_set_dir_pca[direction]['data'].shape))
                data_set_dir_pca[direction]['pcaComponentsCount'] = pandas.DataFrame(
                    dim_reduction_estimator.components_,
                    index=data_set_dir_pca[direction]['data'].columns,
                    columns=original_columns
                )

                if directed_data_set[direction].get('validation', None) is not None:
                    data_set_dir_pca[direction]['validation'] = pandas.DataFrame(
                        dim_reduction_estimator.transform(directed_data_set[direction]['validation'].values),
                        index=directed_data_set[direction]['validation'].index
                    )
                    data_set_dir_pca[direction]['validation'].columns = pandas.Index(
                        ['pca_' + str(i) for i in data_set_dir_pca[direction]['validation'].columns]
                    )

                else:
                    data_set_dir_pca[direction]['validation'] = None

            elif key == 'validation' and data_set_dir_pca[direction].get('validation') is None:
                data_set_dir_pca[direction]['validation'] = None

            else:
                data_set_dir_pca[direction][key] = directed_data_set[direction][key]

    return data_set_dir_pca
