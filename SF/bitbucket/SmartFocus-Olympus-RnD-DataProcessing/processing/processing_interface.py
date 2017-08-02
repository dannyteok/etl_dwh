from . import constants
from . import parsers as pr
from .processors import static_data_processing as sdpr
from .processors import temporal_data_processing as tdpr
import hashlib
import numpy as np
import pandas
import copy


def process_fingerprints(*,raw_data_df:pandas.DataFrame, **kwargs)->dict:
    """ process raw fingerprint data to create fingerprints
        Static fingerprints assumes that all beacons are in the setup
        validation fingerprints do not and use the beacons in the configuration
    Args:
        raw_data_df: a dataframe with the raw data
        **kwargs:
            workers the number of workers to use for the validation trainig
            configuration: An instance of the ProcessingConfiguration class
    Returns:
        dictionary with the processed data

    """
    processed_data = kwargs.pop("processed_data", None)
    validation_raw_data = kwargs.pop("validation_raw_data", pandas.DataFrame())
    debug = kwargs.pop("debug",False)
    workers = 1 if debug else kwargs.pop("workers", 8)

    configuration = kwargs.pop("configuration")

    statistics = sdpr.create_statistics_from_raw_data(
        raw_ble_data_df=raw_data_df,
        log_data=configuration.log_data,
    )
    original_statistics = copy.deepcopy(statistics)

    if processed_data is None:
        print ("statistics:\n{}\n".format(statistics.describe()))

        features_df = sdpr.create_features_from_statistics(
            configuration=configuration,
            statistics_df=statistics,
        )

        processed_data = pr.process_features_data(
            data_features=features_df,
            pcaComponentsCount=configuration.pca_components_count,
            missing_value=configuration.missing_value,
            features=configuration.features
        )

    processed_data['statistics'] = original_statistics

    processed_data["settings"] = {
        "pcaComponentsCount": configuration.pca_components_count,
        "window_size": configuration.window_size,
        "window_step": configuration.window_step,
        "beacons": configuration.beacons,
        "dropObs": configuration.drop_low_observations,
        "missing_value": configuration.missing_value,
        "observed_value": configuration.observed_value,
        "features": configuration.features,
        "filter_beacons": configuration.filter_beacons,
        "channel": configuration.channel,
        'validation_type': configuration.validation_type,
        'frequency_threshold':configuration.frequency_threshold,
        "nodes":len(statistics.index.get_level_values("node").unique())
    }

    for direction in ['N', 'E', 'S', 'W']:
        beacons = set(processed_data[direction]['fingerprints'].columns.get_level_values('beacon').unique())
        if configuration.filter_beacons is not None:
            expected_beacons = set(
                processed_data['settings']['beacons']) - set(configuration.filter_beacons)
        else:
            expected_beacons = set(
                processed_data['settings']['beacons'])
        assert expected_beacons == beacons, 'beacons are wrong with the saved beacons'

    print("processing {} data".format(configuration.validation_type))
    window_features = process_validation_data(raw_data_df=raw_data_df,
                                                  processed_data=processed_data,
                                                  workers=workers,
                                                  configuration=configuration,
                                                  )


    validation_features = process_validation_data(raw_data_df=validation_raw_data,
                                                  processed_data=processed_data,
                                                  workers=workers,
                                                  configuration=configuration
                                                 )

    for direction in constants.DIRECTIONS:
        assert processed_data[direction]['data'].shape[1] == window_features[direction].shape[1]
        processed_data[direction]['window'] = window_features[direction]

        if validation_features:
            processed_data[direction]['validation'] = validation_features[direction]
        else:
            processed_data[direction]['validation'] = window_features[direction]

    if 'obs' in configuration.features:
        configuration.features.remove('obs')

    processed_data['identifier'] = configuration.identifier

    node_string = ".".join(processed_data['N']['data'].index.get_level_values("node"))
    processed_data['node_identifier'] = hashlib.sha1(node_string.encode('utf-8')).hexdigest()

    return processed_data


def process_validation_data(*,raw_data_df:pandas.DataFrame, processed_data:dict, **kwargs)->dict:
    """

    Args:
        raw_data_df: the raw data to use for validation
        processed_data: processed data to use norm and pca transform
        **kwargs:
            window_size = kwargs.pop("window_size")
            window_step = kwargs.pop("window_step")
            beacons = kwargs.pop("beacons", raw_data_df['beacon'].unique())
            drop_low_observations = kwargs.pop("drop_low_observations")
            missing_value = kwargs.pop("missing_value")
            observed_value = kwargs.pop("observed_value")
            features = kwargs.pop("features")
            filter_beacons = kwargs.pop("filter_beacons",None)
            channel = kwargs.pop("channel")
            workers = kwargs.pop("workers", 8)

    Returns:

    """
    if raw_data_df.size == 0: return {}

    configuration = kwargs.pop("configuration", None)
    if configuration.log_data:
        raw_data_df['rssi'] = np.log(-raw_data_df['rssi'])

    workers = kwargs.pop("workers", 8)

    rolled_features = tdpr.create_features_from_raw_data(
        raw_data_df,
        configuration=configuration,
        workers=workers,
    )
    print ("rolled data description:\n{}\n".format(rolled_features.unstack("stat").stack("window").describe()))
    validation_features = pr.transform_temporal_features(
        rolled_features,
        processed_data,
        channel=configuration.channel,
        missing_value=configuration.missing_value,
    )

    return validation_features
