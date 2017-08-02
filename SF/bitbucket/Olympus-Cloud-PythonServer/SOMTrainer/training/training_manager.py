#! /usr/bin/env python
import copy
import itertools
import os
import pprint
import time
import joblib
import numpy as np
import threading
import json
import processing.io_interface as io_interface
import processing.processing_interface as processing_interface
from SOMTrainer.serverio.validation import save_report_pickle
from SOMTrainer.validation.som_validator import som_validator, create_som_report
from SOMTrainer.model.model_interface import save_model
import SOMTrainer.training.som_trainer as som_trainer
import requests
# import SOMTrainer.visualization.tools as viztools
from SOMTrainer.utils import get_path
from processing.configuration import ProcessingConfiguration


class JobManager(object):
    def __init__(self):
        self.channel_list = ['^25', '^26', '^27']
        self.direction_list = ['N', 'E', 'S', 'W']

    def job_producer(self, configuration):

        parameters = copy.deepcopy(configuration)
        parameters['method'] = parameters.get('method', 'pickle')
        nodesOptimizer = configuration.get('nodesOptimizer', None)
        if nodesOptimizer:
            nodesIterator = range(30, 55, 5)

        else:
            nodesIterator = [int(configuration['somSize'])]

        # somtypeIterator = ['emsom', 'som']
        somtypeIterator = ['tree']
        simulations = int(configuration.get('simulations_count', 1))

        simulationsIterator = range(simulations)

        channel_list = []

        use_ios = bool(configuration.get('doIOS', True))
        if use_ios:
            channel_list.append('0')
        use_android = bool(configuration.get('doAndroid', True))

        if use_android:
            channel_list.extend(['25', '26', '27'])

        if len(channel_list) == 0:
            channel_list = ['0', '25', '26', '27']

        savedir = get_path(configuration, 'fingerprints/processed')

        for job_configuration in itertools.product(simulationsIterator, nodesIterator, somtypeIterator,
                                                   channel_list, self.direction_list):

            parameters2 = parameters.copy()
            parameters2['somSize'] = int(job_configuration[1])
            parameters2['som_type'] = job_configuration[2]
            parameters2['channel'] = job_configuration[3]
            parameters2['direction'] = job_configuration[4]

            if parameters2['channel'] == '0':
                operating_system = 'ios'
            else:
                operating_system = 'android'

            filename = os.path.join(savedir,
                                    'training-processed-data-{}-{}-{}.pkl'.format(
                                        operating_system,
                                        parameters2['channel'],
                                        parameters2['pcaComponentsCount']

                                    ))
            parameters2['data_filename'] = filename
            # print("simulations: {}, somSize: {} , type: {}, channel: {}, direction:{}".format(
            #     job_configuration[0],
            #     job_configuration[1],
            #     job_configuration[2],
            #     job_configuration[3],
            #     job_configuration[4]
            # ))
            yield parameters2

    def prepare_data(self, configuration_json, debug=False):

        assert configuration_json.get("beacons") is not None
        config = ProcessingConfiguration.create_configuration(dictionary=configuration_json)
        print("configuration:\n{}".format(config))
        raw_data_df = io_interface.load_raw_fingerprint_data(configuration=config)
        validation_raw_df = io_interface.load_raw_fingerprint_data(configuration=config, suffix='validation/validation_fingerprints')
        raw_data_beacons = raw_data_df['beacon'].unique().tolist()
        assert set(raw_data_beacons) == set(config.beacons), "please add raw data beacons to config file: {}".format(raw_data_beacons)

        print("nodes fingerprinted: {}".format(raw_data_df['node'].unique().size))
        if validation_raw_df.size >0: print("found validation fingerprints: {}".format(validation_raw_df['node'].unique().size))

        if config.beacons is None:
            config.beacons = raw_data_df.unique().tolist()
            if debug: print("using beacons from raw data:\n{}".format(config.beacons))

        channel_list = []
        if configuration_json['doIOS']:
            channel_list.append('0')

        if configuration_json['doAndroid']:
            channel_list.extend(['25', '26', '27'])

        for channel in channel_list:
            config.channel = channel
            if config.check_processed_file_exists() and not debug:
                print("file exists")
            else:
                processed_data = processing_interface.process_fingerprints(
                    raw_data_df=raw_data_df,
                    validation_raw_data=validation_raw_df,
                    configuration=config,
                    debug=debug
                )
                io_interface.save_processed_fingerpint_data(processed_data, config=config)
        print ("completed data wrangling")


    def train_data_parallel(self, configuration_json, debug=False):
        nodes = configuration_json['somSize']

        print("nodes per edge number: {}".format(nodes))

        iterations = int(configuration_json['iterations'])
        ios_training = bool(configuration_json.get('doIOS', False))
        android_training = bool(configuration_json.get('doAndroid', False))
        simulations_count = configuration_json.get("simulations_count", 1)
        simulations_done = 4 * (int(ios_training) + 3 * int(android_training)) * simulations_count
        max_iterations = int(configuration_json['maxIterations'])
        print("iterations: {} max: {}".format(iterations, max_iterations))
        workers = 1 if debug else int(configuration_json.get('workers', 6))
        verbose = 10 if simulations_done < 12 else 100
        print ('workers: {}'.format(workers))

        start_time = time.time()
        with joblib.Parallel(n_jobs=workers, backend='multiprocessing', verbose=verbose,
                             pre_dispatch='2.0*n_jobs') as parallel:
            results = parallel(
                joblib.delayed(som_trainer.som_trainer, check_pickle=False)(job_configuration) for job_configuration in
                self.job_producer(configuration_json))
            assert len(results) == simulations_done, 'simulations done: {}, expected {} '.format(len(results),
                                                                                                 simulations_done)
        end_time = time.time()

        print(
            "time taken: {} minutes , mean error: {}".format((end_time - start_time) / 60.,
                                                             np.nanmean(np.asarray(results))))

    def json_model_creation(self, configuration_json, add_empty_weights=True, legacy_json=False):

        precision = configuration_json['precision']
        root_path = get_path(configuration_json)
        som_folder = configuration_json.get('som_folder','somoutput')
        trained_som_path = os.path.join(root_path, 'fingerprints', som_folder)
        input_json = os.path.join(root_path, 'model/empty_model.json')
        optimal_data_frame_path = os.path.join(root_path, 'fingerprints/validation/som_report.pkl')

        optimal_dataframe = joblib.load(optimal_data_frame_path) if os.path.exists(optimal_data_frame_path) else None

        configuration = ProcessingConfiguration.create_configuration(configuration_json)
        if configuration_json['doIOS']:
            channel_list = ['0']
            output_json = os.path.join(root_path, 'model/trained_model_ios.json')

            save_model(input_json=input_json,
                       output_json=output_json,
                       trained_som_path=trained_som_path,
                       channel_list=channel_list,
                       configuration=configuration,
                       precision=precision,
                       optimal_dataframe=optimal_dataframe,
                       add_empty_weights=add_empty_weights,
                       legacy_json=legacy_json

                       )
        if configuration_json['doAndroid']:
            channel_list = ['25', '26', '27']
            output_json = os.path.join(root_path, 'model/trained_model_android.json')
            save_model(input_json=input_json,
                       output_json=output_json,
                       trained_som_path=trained_som_path,
                       channel_list=channel_list,
                       configuration=configuration,
                       precision=precision,
                       optimal_dataframe=optimal_dataframe,
                       add_empty_weights=add_empty_weights,
                       legacy_json=legacy_json
                       )
        print ("json created")

    def optimize_data_parallel(self, configuration_json, debug=False):
        print ("starting validation")
        somSize = int(configuration_json['somSize'])
        print("somSize per edge number: {}".format(somSize))

        ios_training = bool(configuration_json.get('doIOS', False))
        android_training = bool(configuration_json.get('doAndroid', False))
        simulations_count = configuration_json.get('simulations_count', 1)
        simulations_done = 4 * (int(ios_training) + 3 * int(android_training)) * simulations_count
        workers = 1 if debug else int(configuration_json.get('workers', 6))
        print ('workers: {}'.format(workers))
        start_time = time.time()
        with joblib.Parallel(n_jobs=workers, backend='multiprocessing', verbose=0,
                             pre_dispatch='2.0*n_jobs') as parallel:
            results = parallel(
                joblib.delayed(som_validator, check_pickle=False)(job_configuration) for job_configuration in
                self.job_producer(configuration_json))
            assert len(results) == simulations_done, 'simulations done: {}, expected {} '.format(len(results),
                                                                                                 simulations_done)

        file_path = get_path(configuration_json, "fingerprints/validation/validation_results.pkl")
        joblib.dump(results, file_path, compress=True)
        end_time = time.time()
        som_report = create_som_report(results)
        save_report_pickle(som_report, configuration_json)

        print(
            "time taken: {} minutes , som report:\n".format(
                (end_time - start_time) / 60.
            )
        )
        pprint.pprint(som_report)

        return som_report

    def full_training(self, configuration_json):
        thread_name = threading.currentThread().name
        threading.currentThread().setName("MainThread")
        configuration_json['data_path'] = '/sites'
        configuration_json['method'] = 'pickle'
        job_id = configuration_json['job']
        self.prepare_data(configuration_json=configuration_json)
        print("{} : starting model training".format(job_id))
        self.train_data_parallel(configuration_json=configuration_json, debug=True)
        print("{} : starting model validation".format(job_id))
        self.optimize_data_parallel(configuration_json=configuration_json)
        print("{} : starting json model creation".format(job_id))
        self.json_model_creation(configuration_json=configuration_json)
        print("{} : task completed".format(job_id))
        threading.currentThread().setName(thread_name)

        return True

    def post_json_to_server(self, configuration_json):
        url = 'http://localhost:7000/receiver'
        root_path = get_path(configuration_json)
        if configuration_json['doIOS']:
            trained_json = os.path.join(root_path, 'model/trained_model_ios.json')
            print("ios trained model exists: {} \n {}".format(os.path.isfile(trained_json), trained_json),
                  )
            with open(trained_json, 'rb') as json_file:
                r = requests.post(url, data=json.load(json_file))
                print (r.status_code)
        if configuration_json['doAndroid']:
            trained_json = os.path.join(root_path, 'model/trained_model_android.json')
            with open(trained_json, 'rb') as json_file:
                r = requests.post(url, data=json.load(json_file))
                print (r.status_code)

    def save_visualization(self, configuration_json):
        plot_types = configuration_json.get('plot_type', ['coordinates'])
        if not isinstance(plot_types, list):
            plot_types = [plot_types]
        # viztools.save_accuracy_plot(configuration_json, plot_types=plot_types)
