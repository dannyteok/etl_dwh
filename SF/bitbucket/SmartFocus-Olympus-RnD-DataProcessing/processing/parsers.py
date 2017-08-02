from __future__ import print_function


from processing.processors import static_data_processing as dpr
from processing.processors import temporal_data_processing as tpr
import processing.constants as constants


def transform_temporal_features(rolled_data, processed_data,
                                missing_value=constants.MISSING_BEACON_DATA_VALUE, channel="0"):

    # if channel != '0':
    #     full_beacons = [element for element in full_beacons if re.match('^'+channel,element)]

    result_dict = {}
    directions = rolled_data.index.get_level_values('direction').unique().tolist()
    for direction in directions:
        norm = processed_data[direction]['norm']
        pcaComponentsCount = processed_data[direction].get('pcaComponentsCount')
        result_dict[direction] = tpr.transform_temporal_data_to_features(
            rolled_data,
            direction=direction,
            norm=norm,
            pcaComponentsCount=pcaComponentsCount,
            missing_value=missing_value,
        )

    return result_dict


def process_features_data(data_features, **kwargs):
    """
    Args:
        data_features: the data features
        **kwargs: pcaComponetCount default is 10
                  missing_value for missing observations
                  features to use
    Returns:

    """
    pcaComponentsCount = kwargs.pop("pcaComponentsCount")
    missing_value = kwargs.pop("missing_value")
    features = kwargs.pop("features")

    data_set, directed_data_set = dpr.group_data_set_by_direction(data_features, features=features,
                                                              missing_value=missing_value
                                                              )
    joint_directed_data_set = dpr.data_set_to_joint_distribution(directed_data_set)


    normalized_data_set = dpr.scale_data(joint_directed_data_set)
    pca_data_set = dpr.reduce_dimensionality(normalized_data_set, pcaComponentsCount=pcaComponentsCount)
    assert 'validation' in pca_data_set['N'].keys(), 'no validation key in pca_data'
    return pca_data_set
