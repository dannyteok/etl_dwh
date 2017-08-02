import argparse
import json
import os, re
import sys
import time
import training.training_manager as training_manager
from validation.json_validation import validate_json, conf_json_schema
import pprint
import numpy as np


def purge(dir, pattern):
    """
    remove files in directory that match pattern
    :rtype: void
    """
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def supervisor(configuration_json, **kwargs):
    workers = kwargs.pop('workers')
    montecarlo_simulations = kwargs.pop("montecarlo_simulations")
    debug = kwargs.pop("debug")
    empty_weights = kwargs.pop("empty_weights")
    legacy_json = kwargs.pop("legacy_json")
    prepare = kwargs.pop("prepare")
    train = kwargs.pop("train")
    optimize = kwargs.pop("optimize")
    visualize = kwargs.pop("visualize")
    create_json = kwargs.pop("create_json")

    if validate_json(configuration_json=configuration_json,
                     json_schema=conf_json_schema
                     ):

        configuration_json['workers'] = workers
        configuration_json['simulations_count'] = montecarlo_simulations

        # try:

        if debug:
            np.seterr(all='raise')
        else:
            np.seterr(all='ignore')

        job_manager = training_manager.JobManager()

        start_time = time.time()

        if prepare:
            job_manager.prepare_data(configuration_json=configuration_json, debug=debug)
        if train:
            job_manager.train_data_parallel(configuration_json=configuration_json, debug=debug)
        if optimize:
            job_manager.optimize_data_parallel(configuration_json=configuration_json, debug=debug)
        if visualize:
            job_manager.save_visualization(configuration_json=configuration_json)
        if create_json:
            job_manager.json_model_creation(configuration_json=configuration_json,
                                            add_empty_weights=empty_weights,
                                            legacy_json=legacy_json
                                            )
        end_time = time.time()
        print("time taken: {} minutes".format((end_time - start_time) / 60.))
        # except Exception as err:
        #     print("there was an error:{}".format(err))
        #     return False
        # else:
        return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='heartbeat SOM Trainer')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    group.add_argument('-c', nargs='?', dest='configuration_json', type=str,
                       help="the filepath to the configuration file")
    parser.add_argument('-w', '--workers', dest='workers', default=1, type=int,
                        help='the number of multiprocessing workers')
    parser.add_argument('--mc', '--montecarlo', dest='montecarlo_simulations', default=1, type=int,
                        help='the number of monte carlo simulations to run')

    parser.add_argument("-p", "--prepare", action="store_true", dest="prepare")
    parser.add_argument("-v", "--visualize", action="store_true", dest="visualize")
    parser.add_argument("-t", "--train", action="store_true", dest="train")
    parser.add_argument("-o", "--optimize", action="store_true", dest="optimize")
    parser.add_argument("-d", "--debug", action="store_true", dest="debug")
    parser.add_argument("-j", "--json", action="store_true", dest="create_json")
    parser.add_argument("-e", "--empty", action="store_false", dest="empty_weights",
                        help="Do not Store Empty Weights in soms")
    parser.add_argument("-l", "--legacy", action="store_true", dest="legacy_json", help="Store legacy json")

    args = parser.parse_args()

    if args.configuration_json:
        with open(args.configuration_json, 'r') as configuration_file:
            configuration_json = json.load(configuration_file)
    else:

        json_string = ""
        for line in args.infile.readlines():
            json_string += line

        configuration_json = json.loads(json_string)

    pprint.pprint(configuration_json)

    check_valid = input("are you sure you want to continue: y/N\n")

    if check_valid.lower() == 'y' or check_valid.lower() == "yes":

        supervisor(configuration_json=configuration_json,
             workers=args.workers,
             montecarlo_simulations=args.montecarlo_simulations,
             debug=args.debug,
             empty_weights=args.empty_weights,
             legacy_json=args.legacy_json,
             prepare=args.prepare,
             train=args.train,
             optimize=args.optimize,
             visualize=args.visualize,
             create_json=args.create_json
             )
    else:
        print("please change your configuration json and try again")
