import ipdb
import processing.constants as constants
import numpy as np

def fill_low_observations(input_df, **kwargs):
    """

    Args:
        input_df: the dataframe to remove low observations count
        **kwargs:
            missing_value = value to use for missing data
            obs_threshold = thershold below which drop data
            frequency_threshold = frequency below which drop data
            observed_value = value to add to missing_value for data below frequency_threshold but above obs_threshold

    Returns:
       dataframe with  missing values or below threshold values fixed
    """
    missing_value = kwargs.pop("missing_value")
    obs_threshold = kwargs.pop("fill_low_obs_threshold")
    frequency_threshold = kwargs.pop("frequency_threshold")
    observed_value = kwargs.pop("observed_value")
    assert frequency_threshold is not None, "frequncy_threshold was None"
    try:
        drop_index = input_df['frequency'] <= frequency_threshold
        keep_index = np.logical_and(input_df['obs'] < obs_threshold, input_df['frequency']>frequency_threshold)
    except:
        ipdb.set_trace()
        raise Exception("input does not have observation number or frequency")
    else:
        for feature in ['min','max','median','mean','quant3']:
            if feature in input_df.columns:
                if "window" in input_df.columns.names:
                    for window in input_df.columns.get_level_values('window'):
                        input_df.loc[drop_index[window], (feature, window)] = missing_value
                        input_df.loc[keep_index[window], (feature, window)] = min(observed_value, -100)
                else:
                    input_df.loc[drop_index, feature] = missing_value
                    input_df.loc[keep_index, feature] = min(observed_value, -100)

        for feature in ['std','mad']:
            if feature in input_df.columns:
                if "window" in input_df.columns.names:
                    for window in input_df.columns.get_level_values('window'):
                        input_df.loc[drop_index[window], (feature, window)] = 0
                        input_df.loc[keep_index[window], (feature, window)] = 0
                else:
                    input_df.loc[drop_index, feature] = 0
                    input_df.loc[keep_index, feature] = 0
    return input_df
