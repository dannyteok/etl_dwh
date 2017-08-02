from __future__ import print_function

import glob
import os

import joblib
import model.json_trainer as jsontr
import processing.io_interface as io_interface
import json
import SOMTrainer.validation.json_validation as model_validation
import ipdb

# Server now picks best regions from trained soms and creates new som

def find_pkl(trained_som_path, som_type, channel, direction,
             nodes="*", components="*", identity=None):
    if channel == '0':
        operating_system = 'ios'
    else:
        operating_system = 'android'

    if identity is None:

        search_string = 'trained-{}-{}-{}-{}nodes-{}dir-{}comp-*.pkl'.format(
            som_type, operating_system, channel, nodes,
            direction, components,
        )

    else:

        search_string = 'trained-{}-{}-{}-{}nodes-{}dir-{}comp-{}*.pkl'.format(
            som_type, operating_system, channel, nodes,
            direction, components, identity
        )

    search_pattern = os.path.join(trained_som_path, search_string)
    found_files = glob.glob(search_pattern)
    if len(found_files) > 0:
        return joblib.load(found_files[0])
    return None


def get_som_file_identity(optimal_dataframe, som_type, channel, direction):
    if optimal_dataframe is not None:
        dataframe =  optimal_dataframe.xs([direction, channel, som_type])
        identities=dataframe['id'].values
        if len(identities)> 1:
            return dataframe.loc[dataframe['error'].argmin()].loc['id']
        else:
            return identities[0]
    return None


def create_trained_model_json(**kwargs):
    trained_som_path = kwargs.pop("trained_som_path")
    processed_data = kwargs.pop("processed_data")
    precision = kwargs.pop("precision")
    channel = kwargs.pop("channel")
    optimal_dataframe = kwargs.pop("optimal_dataframe")
    add_empty_weights = kwargs.pop("add_empty_weights")
    legacy_json = kwargs.pop("legacy_json")

    direction_list = ['N', 'E', 'S', 'W']
    joint_list = []
    som_list = []
    for som_type in ['tree']:
        for direction in direction_list:
            identity = get_som_file_identity(optimal_dataframe,
                                             som_type=som_type,
                                             channel=channel,
                                             direction=direction
                                             )
            som_dictionary = find_pkl(
                trained_som_path=trained_som_path,
                som_type=som_type,
                channel=channel,
                direction=direction,
                identity=identity)
            if som_dictionary is not None:
                joint_data = processed_data[direction]['joint']
                joint_dictionary = jsontr.transform_joint_distribution_to_json(
                    joint=joint_data,
                    channel=channel,
                    direction=direction,
                    precision=precision)

                joint_list.append(joint_dictionary)
                som_json = jsontr.transform_som_dictionary_to_json(
                    som_dictionary=som_dictionary,
                    norm=processed_data[direction]['norm'],
                    components=processed_data[direction].get('pcaComponentsCount'),
                    precision=precision,
                    add_empty_weights=add_empty_weights,
                    legacy_json=legacy_json

                )
                if legacy_json:
                    som_list.append(som_json)
                else:
                    som_list.append({"innerSom": som_json})
    return joint_list, som_list


def save_model(**kwargs):
    """
    Args:
        **kwargs:
            input_json = kwargs.pop("input_json")
            output_json = kwargs.pop("output_json")
            trained_som_path = kwargs.pop("trained_som_path")
            channel_list = kwargs.pop("channel_list")
            precision = kwargs.pop("precision")
            configuration = kwargs.pop("configuration")
            optimal_dataframe = kwargs.pop("optimal_dataframe")
            add_empty_weights = kwargs.pop("add_empty_weights")
    """
    input_json = kwargs.pop("input_json")
    output_json = kwargs.pop("output_json")
    trained_som_path = kwargs.pop("trained_som_path")
    channel_list = kwargs.pop("channel_list")
    precision = kwargs.pop("precision")
    configuration = kwargs.pop("configuration")
    optimal_dataframe = kwargs.pop("optimal_dataframe")
    add_empty_weights = kwargs.pop("add_empty_weights")
    legacy_json = kwargs.pop("legacy_json")

    with open(input_json, 'r') as input_file:
        empty_json = json.load(input_file)
        joint_list = []
        som_list = []

        for channel in channel_list:
            configuration.channel = channel
            processed_data = io_interface.load_processed_fingerpint_data(configuration)

            joint_json, som_json = create_trained_model_json(
                trained_som_path=trained_som_path,
                processed_data=processed_data,
                precision=precision,
                channel=channel,
                optimal_dataframe=optimal_dataframe,
                add_empty_weights=add_empty_weights,
                legacy_json=legacy_json
            )
            joint_list.extend(joint_json)
            som_list.extend(som_json)

        empty_json['jointDistributions'] = joint_list
        empty_json['som'] = som_list

        with open(output_json, 'w') as trained_json_file:
            model_validation.validate_json(empty_json, model_validation.model_json_schema)
            json.dump(empty_json, trained_json_file, sort_keys=True)
