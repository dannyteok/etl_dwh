from behave import *
import pandas
import processing.io_interface as io
import processing.processing_interface as pr
import processing.processors.static_data_processing as sdpr
import shutil
import os
import re
import numpy as np


def check_validation_data(validation_data):
    assert isinstance(validation_data, dict)
    assert len(validation_data.keys()) == 4
    for key in validation_data.keys():
        assert key in ['N', 'E', 'S', 'W']
        assert all(x in validation_data[key].index.names for x in [u'node', u'direction'])
        assert validation_data[key].index.get_level_values('node').size >= validation_data[key].index.get_level_values(
            'node').unique().size


def check_processed_rolled_data(processed_rolled_data, full_beacons):
    assert isinstance(processed_rolled_data, pandas.DataFrame)
    assert processed_rolled_data.columns.name == 'window'
    assert all(x in processed_rolled_data.index.names for x in
               [u'node', u'direction', u'stat', u'beacon']), "index\n:{}".format(processed_rolled_data.index)
    assert all(x in processed_rolled_data.index.get_level_values('direction') for x in ['N', 'E', 'S', 'W'])
    assert full_beacons.issuperset(
        set(processed_rolled_data.index.get_level_values('beacon'))), "beacons don't belong in all beacons"


def check_processed_fingerprint_data(processed_data, channel, pcaComponentsCount):
    assert isinstance(processed_data, dict)
    assert len(processed_data.keys()) == 7, "keys are not 7: {}".format(processed_data.keys())
    assert all(x in processed_data.keys() for x in ['N', 'E', 'S', 'W'])
    for key in processed_data.keys():
        assert all(x in processed_data[key].keys() for x in ['joint', 'norm', 'data', 'validation'])
        beacons = processed_data[key]['joint'][0].index.get_level_values(level='beacon')
        if channel == '0':
            regexes = [
                re.compile('^25'),
                re.compile('^26'),
                re.compile('^27'),
            ]
        else:
            regexes = [
                re.compile('^' + channel)
            ]
        assert all(any(regex.match(beacon) for regex in regexes) for beacon in
                   beacons), 'joint beacons not of correct channel {}'.format(beacons)
        norm_beacons = processed_data[key]['norm'][0].index.get_level_values(level='beacon')
        norm2_beacons = processed_data[key]['norm'][0].index.get_level_values(level='beacon')
        assert all(any(regex.match(beacon) for regex in regexes) for beacon in
                   norm_beacons), 'norm mean beacons not of correct channel {}'.format(beacons)
        assert all(any(regex.match(beacon) for regex in regexes) for beacon in
                   norm2_beacons), 'norm std beacons not of correct channel {}'.format(beacons)
        assert processed_data[key]['norm'][0].size == processed_data[key]['norm'][
            1].size, 'norm mean not equal to norm std'
        assert processed_data[key]['joint'][0].size == processed_data[key]['norm'][1].size, 'joint not equal to joint'
        assert isinstance(processed_data[key]['data'], pandas.DataFrame)
        assert 'node' in processed_data[key]['data'].index.names
        assert 'direction' in processed_data[key]['data'].index.names

        if pcaComponentsCount > 1:
            assert processed_data[key]['data'].shape[
                       1] == pcaComponentsCount, 'number of pcaComponentsCount in data is not correct: {} {}'.format(
                pcaComponentsCount,
                processed_data[
                    key]['data'])
            assert isinstance(processed_data[key]['pcaComponentsCount'], pandas.DataFrame)
            assert processed_data[key]['pcaComponentsCount'].shape[
                       0] == pcaComponentsCount, 'number of pcaComponentsCount in processed pcaComponentsCount is not correct: {} {}'.format(
                pcaComponentsCount, processed_data[key]['pcaComponentsCount'])
            assert processed_data[key]['pcaComponentsCount'].shape[1] == processed_data[key]['joint'][0].size

        elif pcaComponentsCount > 0:
            assert processed_data[key]['data'].shape[1] < processed_data[key]['norm'][
                0].size, 'number of beacons and stats does not max norm'
            assert isinstance(processed_data[key]['pcaComponentsCount'], pandas.DataFrame)
            assert processed_data[key]['pcaComponentsCount'].shape[
                       0] == pcaComponentsCount, 'number of pcaComponentsCount is not correct'
            assert processed_data[key]['pcaComponentsCount'].shape[1] == processed_data[key]['joint'][0].size
            # assert all(x in processed_data[key ]['data'].columns.names for x in ['stat','beacon'])
        else:
            assert all(x in processed_data[key]['data'].columns.names for x in ['stat', 'beacon'])
            assert processed_data[key]['data'].shape[1] == processed_data[key]['norm'][
                1].size, 'data feature size not equal to norm'


@given("the default test configuration json file with channel {channel}")
def step_impl(context, channel):
    context.configuration_json = {
        "method": 'csv',
        "useMax": True,
        "data_path": context.rootpath,
        "owner": "SMARTFOCUS", "channel": channel,
        "split_point": 1.0, "useMad": False,
        "floor": "Floor8", "site": "London", "useMedian": True,
        "features": ['max', 'median'],
        "pcaComponentsCount": 0,
        'window_size': 10,
        'window_step': 2,
        "dropObs":False,
        "observed_value":5,
        "missing_value":-105

    }

    processed_path = os.path.join(context.rootpath,
                                  context.configuration_json['owner'],
                                  context.configuration_json['site'],
                                  context.configuration_json['floor'],
                                  "fingerprints/processed")
    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.mkdir(processed_path)
    validation_path = os.path.join(context.rootpath,
                                   context.configuration_json['owner'],
                                   context.configuration_json['site'],
                                   context.configuration_json['floor'],
                                   "fingerprints/validation")

    if os.path.exists(validation_path):
        shutil.rmtree(validation_path)
    os.mkdir(validation_path)


@given("the configuration json file with channel {channel} and features median max frequency")
def step_impl(context, channel):
    context.configuration_json = {
        "method": 'csv',
        "useMax": True,
        "data_path": context.rootpath,
        "owner": "SMARTFOCUS", "channel": channel,
        "split_point": 1.0, "useMad": False,
        "floor": "Floor1", "site": "Brighton", "useMedian": True,
        "features": ["median", "max", "frequency"]
    }
    processed_path = os.path.join(context.rootpath,
                                  context.configuration_json['owner'],
                                  context.configuration_json['site'],
                                  context.configuration_json['floor'],
                                  "fingerprints/processed")
    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.mkdir(processed_path)
    validation_path = os.path.join(context.rootpath,
                                   context.configuration_json['owner'],
                                   context.configuration_json['site'],
                                   context.configuration_json['floor'],
                                   "fingerprints/validation")

    if os.path.exists(validation_path):
        shutil.rmtree(validation_path)
    os.mkdir(validation_path)


@given("the Brighton configuration json file with channel {channel}")
def step_impl(context, channel):
    context.configuration_json = {
        "method": 'csv',
        "useMax": True,
        "data_path": context.rootpath,
        "owner": "SMARTFOCUS", "channel": channel,
        "split_point": 1.0, "useMad": False,
        "floor": "Floor1", "site": "Brighton", "useMedian": True,
    }
    processed_path = os.path.join(context.rootpath,
                                  context.configuration_json['owner'],
                                  context.configuration_json['site'],
                                  context.configuration_json['floor'],
                                  "fingerprints/processed")
    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.mkdir(processed_path)
    validation_path = os.path.join(context.rootpath,
                                   context.configuration_json['owner'],
                                   context.configuration_json['site'],
                                   context.configuration_json['floor'],
                                   "fingerprints/validation")

    if os.path.exists(validation_path):
        shutil.rmtree(validation_path)
    os.mkdir(validation_path)


@when("the raw fingerprints are loaded from {method}")
def step_impl(context, method):
    raw_data_frame = io.load_raw_fingerprint_data(configuration_json=context.configuration_json)
    assert isinstance(raw_data_frame, pandas.DataFrame), 'raw data is dataframe'
    assert all([x in raw_data_frame.columns for x in [u'timestamp', u'node', u'beacon', u'rssi', u'direction', u'set']])
    assert not np.any(raw_data_frame.isnull().values), 'all data is valid'
    context.raw_data = raw_data_frame

    # context.full_beacons = set(context.data[3]['beacon'])
    #
    # if context.configuration_json.get('filter_beacons') is not None:
    #     assert all(x not in context.full_beacons for x in context.configuration_json['filter_beacons'])
    #
    # assert len(context.data) == 4
    # assert isinstance(context.data[0], pandas.DataFrame)
    # assert context.data[0].size > 0
    # assert isinstance(context.data[1], pandas.DataFrame)
    # assert context.data[1].size > 0
    # assert isinstance(context.data[3], pandas.DataFrame)
    # assert context.data[3].size > 0
    # if context.configuration_json['split_point'] == 1:
    #     assert np.all(context.data[2].isnull()), "{}".format(context.data[2])
    # else:
    #     assert isinstance(context.data[2], pandas.DataFrame)


@when("the raw fingerpints are loaded from {method} with a few filtered beacons")
def step_impl(context, method):
    context.configuration_json['method'] = method
    context.configuration_json['filter_beacons'] = ['2600E3E4000B', '2505A65C000B', '2700E462000B']
    context.execute_steps(u'''
         when the raw fingerprints are loaded from {}
    '''.format(method))


@then("Processed data is loaded from {method}")
def step_impl(context, method):
    context.configuration_json['method'] = method
    context.processed_data = io.load_processed_fingerpint_data(context.configuration_json)
    check_processed_fingerprint_data(processed_data=context.processed_data,
                                     channel=context.configuration_json['channel'],
                                     pcaComponentsCount=context.configuration_json['pcaComponentsCount']
                                     )


@then("data is processed with {pcaComponentsCount} pcaComponentsCount")
def step_impl(context, pcaComponentsCount):
    pcaComponentsCount = int(pcaComponentsCount)
    context.configuration_json['pcaComponentsCount'] = pcaComponentsCount
    processed_data = pr.process_fingerprints(context.raw_data, pcaComponentsCount=pcaComponentsCount,
                                             channel=context.configuration_json['channel'],
                                             )
    check_processed_fingerprint_data(processed_data=processed_data,
                                     channel=context.configuration_json['channel'],
                                     pcaComponentsCount=context.configuration_json['pcaComponentsCount']
                                     )
    context.processed_data = processed_data


# @then("raw Data is saved to {method}")
# def step_impl(context, method):
#     context.configuration_json['method'] = method
#     io.save_raw_fingerprint_data(data=context.data,
#                                  configuration_json=context.configuration_json)
#
#     channel = context.configuration_json['channel']
#     if channel == '0':
#         operating_system = 'ios'
#     else:
#         operating_system = 'android'
#     filepath = context.configuration_json['owner'] + "/" + context.configuration_json['site'] + "/" + \
#                context.configuration_json['floor'] + '/fingerprints/processed/training-raw-data-{}-{}.pkl'.format(
#         operating_system, channel)
#     filepath = os.path.join(context.rootpath, filepath)
#     assert os.path.isfile(filepath), \
#         ' file does not exit {}'.format(filepath)


@then("processed Data is saved to {method}")
def step_impl(context, method):
    context.configuration_json['method'] = method
    io.save_processed_fingerpint_data(data=context.processed_data,
                                      configuration_json=context.configuration_json)
    channel = context.configuration_json['channel']
    pcaComponentsCount = context.configuration_json['pcaComponentsCount']
    if channel == '0':
        operating_system = 'ios'
    else:
        operating_system = 'android'

    filepath = context.configuration_json['owner'] + "/" + context.configuration_json['site'] + "/" + \
               context.configuration_json[
                   'floor'] + '/fingerprints/processed/training-processed-data-{}-{}-{}.pkl'.format(
        operating_system, channel, pcaComponentsCount)
    filepath = os.path.join(context.rootpath, filepath)
    assert os.path.isfile(filepath), \
        ' file does not exit {}'.format(filepath)


@then("Rolled Data is processed with window size: {window_size} and step: {window_step}")
def step_impl(context, window_size, window_step):
    window_size = int(window_size)
    window_step = int(window_step)
    # pcaComponentsCount = float(pcaComponentsCount)
    # context.configuration_json['pcaComponentsCount'] = pcaComponentsCount
    processed_rolled_data = pr.process_rolling_data(context.data,
                                                    window_size=window_size,
                                                    window_step=window_step,
                                                    features=context.configuration_json.get(['features'],
                                                                                            ['max', 'median'])
                                                    )

    check_processed_rolled_data(processed_rolled_data, context.full_beacons)

    context.processed_rolled_data = processed_rolled_data


@then("Rolled Data is saved to {method}")
def step_impl(context, method):
    context.configuration_json['method'] = method
    io.save_rolled_data(context.processed_rolled_data,
                        configuration_json=context.configuration_json
                        )

    channel = context.configuration_json['channel']
    if channel == '0':
        operating_system = 'ios'
    else:
        operating_system = 'android'

    filepath = context.configuration_json['owner'] + "/" + context.configuration_json['site'] + "/" + \
               context.configuration_json[
                   'floor'] + '/fingerprints/processed/training-rolled-data-{}-{}.pkl'.format(
        operating_system, channel)
    filepath = os.path.join(context.rootpath, filepath)
    assert os.path.isfile(filepath), \
        ' file does not exit {}'.format(filepath)


@then("Rolled Data is loaded from {method}")
def step_impl(context, method):
    context.configuration_json['method'] = method
    context.processed_rolled_data = io.load_rolled_data(context.configuration_json)
    check_processed_rolled_data(context.processed_rolled_data,
                                context.full_beacons
                                )


@then("Validation Data is created from Rolled Data")
def step_impl(context):
    rolled_data = context.processed_rolled_data
    validation_data = pr.process_validation_data(rolled_data, full_beacons=context.full_beacons,
                                                 processed_data=context.processed_data)
    context.validation_data = validation_data
    check_validation_data(validation_data)


@then("Validation Data is saved to {method}")
def step_impl(context, method):
    context.configuration_json['method'] = method
    io.save_validation_data(context.validation_data, configuration_json=context.configuration_json)

    channel = context.configuration_json['channel']
    pcaComponentsCount = context.configuration_json['pcaComponentsCount']
    if channel == '0':
        operating_system = 'ios'
    else:
        operating_system = 'android'

    filepath = os.path.join(context.rootpath,
                            context.configuration_json['owner'],
                            context.configuration_json['site'],
                            context.configuration_json['floor'],
                            'fingerprints/processed/validation-processed-data-{}-{}-{}.pkl'.format(
                                operating_system, channel, pcaComponentsCount)
                            )

    assert os.path.isfile(filepath), \
        ' file does not exist {}'.format(filepath)


@then("validation data is loaded from {method}")
def step_impl(context, method):
    context.configuration_json['method'] = method
    validation_data = io.load_validation_data(context.configuration_json)
    check_validation_data(validation_data)
    context.validation_data = validation_data


@then("the features are {features}")
def step_impl(context, features):
    # edited_features = features[2:-2].split('\"')[::2]
    edited_features = features.split(',')
    context.configuration_json['features'] = edited_features


@then("the data are filtered {filtered}")
def step_impl(context, filtered):
    context.configuration_json['quant_filter'] = bool(filtered)


@then("the low observations are dropped {drop_low}")
def step_impl(context, drop_low):
    context.configuration_json['dropLowObservations'] = bool(drop_low)


@then("the missing observation value is {missing_value}")
def step_impl(context, missing_value):
    context.configuration_json['missing_value'] = float(missing_value)


@then("the features tuple is created")
def step_impl(context):
    """


                        pcaComponentsCount = kwargs.pop("pcaComponentsCount", 10)
                        window_size = kwargs.pop("window_size",10)
                        window_step = kwargs.pop("window_step",2)
                        filter_data = kwargs.pop("filter_data",False)
                        beacons = kwargs.pop("beacons",raw_data_df['beacon'].unique())
                        drop_low_observations = kwargs.pop("drop_low_observatons",False)
                        missing_value = kwargs.pop("missing_value",constants.MISSING_BEACON_DATA_VALUE)
                        features = kwargs.pop("features",["max", "median"])
                        filter_beacons = kwargs.pop("filter_beacons",None)
                        channel = kwargs.pop("channel","0")


    """
    processed_fingerprints = pr.process_fingerprints(context.raw_data,
                                                     pcaComponentsCount=context.configuration_json[
                                                         'pcaComponentsCount'],
                                                     channel=context.configuration_json['channel'],
                                                     window_size=context.configuration_json["window_size"],
                                                     window_step=context.configuration_json["window_step"],
                                                     filter_data=context.configuration_json.get("filter_data", False),
                                                     # beacons=context.configuration_json.get("beacons"),
                                                     observed_value=context.configuration_json["observed_value"],
                                                     drop_low_observations=context.configuration_json["dropObs"],
                                                     missing_value=context.configuration_json["missing_value"],
                                                     features=context.configuration_json["features"],
                                                     filter_beacons=context.configuration_json.get("filter_beacons"),
                                                     )
    assert isinstance(processed_fingerprints, dict)
    check_processed_fingerprint_data(processed_fingerprints,
                                     context.configuration_json['channel'],
                                     context.configuration_json['pcaComponentsCount']
                                     )
