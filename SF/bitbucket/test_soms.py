import os
import re
import time
import json
from collections import namedtuple
import shutil
import glob
import copy
import argparse
import sys

import pandas
import joblib
import numpy as np
import scipy.spatial.distance as dist
import ipdb

from processing.configuration import ProcessingConfiguration
import processing.io_interface as io_interface
import processing.processors.temporal_data_processing as tdpr
from processing.utils.helpers import get_coordinates
from som.utils.helpers import get_coordinates
from som.soms import tree_som
import processing.parsers as pr
import som.somIO as somio

Tree = namedtuple("Tree", ["tree", "id"])
Simulation = namedtuple("Simulation", ["path", "direction", "coordinates"])
Validation = namedtuple("Validation", ["data", "direction"])


def calculate_validation_per_node(input_data: pandas.DataFrame, name: str, som_dic: dict, workers: int) -> pandas.DataFrame:
    test_points_count = input_data.index.get_level_values('node').unique().size

    soms_count = sum([len(som_dic[i]) for i in som_dic.keys()])

    print("validation points: {}".format(test_points_count))

    print("calculating results estimated tasks: {}".format(soms_count * test_points_count / 4.))

    start_time = time.time()
    results_df = calculate_mean_error_per_node(input_data, som_dic, workers=workers, verbose=10)
    end_time = time.time()

    print("time taken: {}".format((end_time - start_time) / 60.))

    results_filename = os.path.join(configuration.path,
                                    "validation/{}_results_data-{}.pickle".format(name, configuration.identifier))

    results_df.to_pickle(results_filename)
    return results_df


def get_closest_fingerprint_node(node: str, processed_data: dict, direction: str)->str:
    node_coordinates = np.array([[int(i) for i in node.split('-')]])
    training_fingerprint_coordinates = get_coordinates(processed_data[direction]['data'])
    fingerprint_coordinates = training_fingerprint_coordinates[
                              np.argmin(dist.cdist(node_coordinates, training_fingerprint_coordinates)), :]
    closest_node = '-'.join(str(int(i)) for i in np.squeeze(fingerprint_coordinates).tolist())
    return closest_node


def prepare_test_data(simulation: namedtuple, configuration: ProcessingConfiguration, processed_data: dict, workers: int=8)->namedtuple:
    configuration = copy.deepcopy(configuration)
    beacon_data = io_interface.load_raw_validation_data(
        configuration=configuration,
        suffix=simulation.path,
        direction=simulation.direction,
        node=simulation.coordinates,
    )

    rolled_features = tdpr.create_features_from_raw_data(raw_data_frame=beacon_data,
                                                         configuration=configuration,
                                                         workers=workers)

    validation_features = pr.transform_temporal_features(
        rolled_features,
        processed_data,
        channel=configuration.channel,
        missing_value=configuration.missing_value,
    )
    validation_features[simulation.direction] = validation_features[simulation.direction].loc[:,
                                                ~np.any(
                                                    validation_features[simulation.direction].isnull(),
                                                    axis=0)]

    return Validation(validation_features[simulation.direction], simulation.direction)


def load_all_simulations(configuration: ProcessingConfiguration):
    path = configuration.get_path("simulator_data/static")
    prog = re.compile("\d+-\d+")
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            direction = folder.split(" ")[-2]
            coordinates = folder.split(" ")[-1]
            assert direction in ["N", "E", "S", "W"], "direction was {}".format(direction)
            assert prog.match(coordinates) is not None, "coordinates was {}".format(coordinates)
            yield Simulation(path=os.path.join("simulator_data/static", folder),
                             direction=folder.split(" ")[-2],
                             coordinates=folder.split(" ")[-1])


def prepare_all_test_data(configuration: ProcessingConfiguration, processed_data: dict, workers=1):
    all_test_data = [
        prepare_test_data(simulation, configuration=configuration, processed_data=processed_data, workers=workers)
        for simulation in load_all_simulations(configuration)]
    return all_test_data


def prepare_validation_data(configuration,processed_data, workers=1):
    configuration = copy.deepcopy(configuration)
    beacon_data = io_interface.load_raw_fingerprint_data(
        configuration=configuration,
        suffix='validation/validation_fingerprints'
    )

    rolled_features = tdpr.create_features_from_raw_data(raw_data_frame=beacon_data,
                                                         configuration=configuration,
                                                         workers=workers)

    validation_features = pr.transform_temporal_features(
        rolled_features,
        processed_data,
        channel=configuration.channel,
        missing_value=configuration.missing_value,
    )

    return validation_features

def create_tree(dictionary):
    return Tree(dictionary, dictionary["id"])


def prepare_all_soms(configuration, direction,som_path="somoutput"):
    som_trees = [create_tree(som_dic) for som_dic in somio.load_trained_soms(configuration, 'tree', direction, som_path)]
    return som_trees


def get_unique_coordinates(coordinates):
    return pandas.DataFrame(coordinates).drop_duplicates().values

def calculate_results_per_node(validation_data_direction, tree, direction):
    som = tree_som.TreeSom.create_from_dictionary(dictionary=tree.tree)
    true_coordinates = get_coordinates(validation_data_direction)
    true_coordinates_tuple = list(set([(int(row[0]), int(row[1])) for row in true_coordinates.tolist()]))[0]
    predictions = som.predict(validation_data_direction)
    try:
        mean = np.nanmean(predictions, axis=0)
        nans = np.isnan(predictions)[:,0]
        error = dist.euclidean(mean, true_coordinates_tuple) if ~np.any(np.isnan(mean)) else np.nan
    except:
        ipdb.set_trace()
        print(" som {} for coordinates {} there was an error:\n{}".format(tree.id,
                                                                          true_coordinates_tuple,
                                                                          predictions))
        return {"error": np.nan, "nan":np.nan ,"mean": np.nan,
            "som": tree.id, "node": true_coordinates, 'direction': direction}
    else:
        mean = (mean[0], mean[1])
        return {"error": error, "nan":np.mean(nans) ,"mean": mean,
                "som": tree.id, "node": true_coordinates_tuple, 'direction': direction, 'predictions':predictions,
                'true_coordinates':true_coordinates,
                }



def tasks_generator_per_node(validation_data: pandas.DataFrame, som_dic: dict):
    for direction in validation_data.index.get_level_values("direction").unique().tolist():
        for som in som_dic[direction]:
            directed_validation_data = validation_data.xs(direction, level="direction")
            for node in directed_validation_data.index.get_level_values("node").unique():
                yield directed_validation_data.xs(node), som, direction



def calculate_mean_error_per_node(validation_data, som_dic, workers=1, verbose=50):
    with joblib.Parallel(n_jobs=workers, verbose=verbose) as parallel:
        results = parallel(
            joblib.delayed(calculate_results_per_node)(data, som, direction)
            for data, som, direction in tasks_generator_per_node(validation_data, som_dic)
        )
    results_copy = copy.deepcopy(results)
    results_df = pandas.DataFrame(results_copy)
    results_df.to_pickle(os.path.join(configuration.path,
                                      "validation/validation_results-{}.pickle".format(configuration.identifier)))
    return results_df



def get_optimal_soms(results_df):
    mean_results = results_df.groupby(['direction', 'som']).mean().reset_index()
    index = mean_results.groupby("direction")['error'].idxmin()
    optimal_results = mean_results.iloc[index.values,:]
    print(optimal_results)
    return optimal_results

def save_optimal_soms(configuration, optimal_results):
    for identifier in optimal_results['som']:
        files_found = glob.glob(os.path.join(configuration.path, "somoutput","*{}*.pkl".format(identifier)))
        if files_found:
            file_name = os.path.split(files_found[0])[1]
            destination_dir =  os.path.join(configuration.path, "somoutput_optimal")
            if not os.path.isdir(destination_dir):
                os.mkdir(destination_dir)
            shutil.copyfile(files_found[0], os.path.join(destination_dir, file_name))
    return optimal_results


def load_validation_data(configuration, validation_data_file, overwrite=False, frequency_threshold=None):
    if not os.path.isfile(validation_data_file) or overwrite:
        print("file does not exist: {}".format(validation_data_file))
        processed_data = io_interface.load_processed_fingerpint_data(configuration=configuration)
        if frequency_threshold:
            configuration.frequency_threshold = frequency_threshold
        validation_data_dict = prepare_validation_data(configuration, processed_data=processed_data)
        validation_data = pandas.concat([item for item in validation_data_dict.values()], axis=0)
        ipdb.set_trace()
        validation_data.to_pickle(validation_data_file)
    else:
        print("file exists: {}".format(validation_data_file))
        validation_data = pandas.read_pickle(validation_data_file)
    return validation_data


def load_test_data(configuration, test_data_file, overwrite=False, frequency_threshold=None):
    if not os.path.isfile(test_data_file) or overwrite:
        print("file does not exist: {}".format(test_data_file))
        processed_data = io_interface.load_processed_fingerpint_data(configuration=configuration)
        if frequency_threshold:
            configuration.frequency_threshold = frequency_threshold
        test_list = prepare_all_test_data(configuration, processed_data=processed_data)
        test_data = pandas.concat([item.data for item in test_list], axis=0)
        test_data.to_pickle(test_data_file)
    else:
        print("file exists: {}".format(test_data_file))
        test_data = pandas.read_pickle(test_data_file)

    return test_data

def get_optimal_som_statistics(optimal_soms, soms, quantile=0.8):
    results = []
    for row in optimal_soms.iterrows():
        error = []

        som_data = soms.loc[soms['som']==row[1].som]

        for index in som_data.index:
            som_error = np.squeeze(dist.cdist(np.asarray(som_data.loc[index,'node'])[None,:],som_data.loc[index,'predictions'])).tolist()
            error.extend(som_error)

        error = np.asarray(error).flatten()
        nans = np.isnan(error)
        row_result = {'som':row[1].som,
                      'direction':row[1].direction,
                      'mean':np.nanmean(error),
                      'std':np.nanstd(error),
                      'rmse':np.sqrt(np.nanmean(error**2)),
                      'quantile':pandas.Series(error[~nans]).quantile(quantile),
                      'nan': np.mean(nans)}
        results.append(row_result)
    return pandas.DataFrame(results)


def get_optimal_som_statistics_per_node(optimal_soms, soms, quantile=0.8, edge_lenght=0.725):
    results = []
    for row in optimal_soms.iterrows():
        som_data = soms.loc[soms['som']==row[1].som]
        for index in som_data.index:
            error = np.squeeze(dist.cdist(np.asarray(som_data.loc[index,'node'])[None,:],som_data.loc[index,'predictions'])).tolist()


            error = np.asarray(error).flatten()
            nans = np.isnan(error)
            row_result = {
                          'som':row[1].som,
                          'direction':row[1].direction,
                          'mean':np.nanmean(error)*edge_lenght,
                          'std':np.nanstd(error)*edge_lenght,
                          # 'rmse':np.sqrt(np.nanmean(error**2)),
                          'quantile':pandas.Series(error[~nans]).quantile(quantile)*edge_lenght,
                          # 'nan': np.mean(nans),
                          'node':som_data.loc[index,'node']
                          }
            results.append(row_result)
    return pandas.DataFrame(results).set_index(['direction','som','node']).sort_index()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='heartbeat SOM Trainer')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    group.add_argument('-c', nargs='?', dest='configuration_json', type=str,
                       help="the filepath to the configuration file")
    parser.add_argument('-w', '--workers', dest='workers', default=1, type=int,
                        help='the number of multiprocessing workers')

    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite")

    args = parser.parse_args()
    if args.configuration_json:
        with open(args.configuration_json, 'r') as configuration_file:
            configuration_json = json.load(configuration_file)
    else:

        json_string = ""
        for line in args.infile.readlines():
            json_string += line

        configuration_json = json.loads(json_string)
    configuration = ProcessingConfiguration.create_configuration(dictionary=configuration_json)

    print(configuration)

    test_data_file = os.path.join(configuration.path,
                                        "validation/test_data-{}.pickle".format(configuration.identifier))

    validation_data_file = os.path.join(configuration.path,
                                  "validation/validation_data-{}.pickle".format(configuration.identifier))

    test_data = load_test_data(configuration, test_data_file, args.overwrite)
    validation_data = load_validation_data(configuration, validation_data_file, args.overwrite)

    np.seterr(all='raise')
    print ("loading soms")
    start_time = time.time()
    som_dic = {direction: prepare_all_soms(configuration, direction=direction, som_path="somoutput_region_optimal") for direction in ["N", "E", "S", "W"]}
    end_time = time.time()
    soms_count = sum([len(som_dic[i]) for i in som_dic.keys()])
    print("soms: {} time taken:{}".format(soms_count, (end_time - start_time)/60.))

    # test_results_df = calculate_validation_per_node(test_data, 'test', som_dic, args.workers)
    validation_results_df = calculate_validation_per_node(validation_data, 'validation', som_dic, args.workers)
    validation_results_optimal = get_optimal_soms(validation_results_df)
    save_optimal_soms(configuration, validation_results_optimal)
    validation_results_per_node= get_optimal_som_statistics_per_node(validation_results_optimal, validation_results_df)

    # test_results_per_node = get_optimal_som_statistics(validation_results_optimal, test_results_df)

    optimal_soms = []
    for row in validation_results_optimal.iterrows():
        direction =  row[1].direction
        best_som = [som for som in som_dic[direction] if som.id == row[1].som]
        optimal_soms.append(best_som)
