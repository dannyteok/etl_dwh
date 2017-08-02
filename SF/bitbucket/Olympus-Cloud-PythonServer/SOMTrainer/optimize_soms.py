import asyncio
import argparse
import json
import copy
from train_som import supervisor
import itertools
import numpy as np
import sys


async def train_scenario(configuration_json, args):
    print("starting scenario:\n{}".format(configuration_json))
    await training_generator(configuration_json=configuration_json,
                                  args=args,
                                  )


async def training_generator(configuration_json, args):
    result = supervisor(configuration_json=configuration_json,
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
    return result


def create_different_models(configuration_json, kwargs):
    log_scale = kwargs.get("log_scale", False)

    sigma_range = kwargs.get("sigma")
    assert len(sigma_range) == 3
    lambda_range = kwargs.get("regularization")
    assert len(lambda_range) == 3
    learning_rate_range = kwargs.get("learning_rate")
    assert len(learning_rate_range) == 3
    frequency_threshold_range = kwargs.get("frequency_threshold")
    assert len(frequency_threshold_range) == 3
    if log_scale:
        sigma_values = np.logspace(sigma_range[0], sigma_range[1], sigma_range[2])
        lambda_values = np.logspace(lambda_range[0], lambda_range[1], lambda_range[2])
        learning_values = np.logspace(learning_rate_range[0], learning_rate_range[1], learning_rate_range[2])
        frequency_values = np.logspace(frequency_threshold_range[0], frequency_threshold_range[1],
                                       frequency_threshold_range[2])
    else:
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], sigma_range[2])
        lambda_values = np.linspace(lambda_range[0], lambda_range[1], lambda_range[2])
        learning_values = np.linspace(learning_rate_range[0], learning_rate_range[1], learning_rate_range[2])
        frequency_values = np.linspace(frequency_threshold_range[0], frequency_threshold_range[1],
                                       frequency_threshold_range[2])

    for sigma, regularization_parameter, learning, frequency in itertools.product(sigma_values, lambda_values,
                                                                                  learning_values, frequency_values):
        edited_configuration = copy.copy(configuration_json)
        edited_configuration['sigma'] = sigma
        edited_configuration['learning_rate'] = learning
        edited_configuration['regularization_parameter'] = regularization_parameter
        edited_configuration['frequency_threshold'] = frequency
        yield edited_configuration


def event_loop(configuration_json, args, **kwargs):
    loop = asyncio.get_event_loop()
    to_do = [train_scenario(edited_conf_json, args) for edited_conf_json in
             create_different_models(configuration_json, kwargs)]
    event_loop_tasks = asyncio.wait(to_do)
    loop.run_until_complete(event_loop_tasks)
    loop.close()


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

    event_loop(configuration_json=configuration_json, args=args,
               sigma=[1.0, 2.0, 2],
               learning_rate=[0.1, 0.5, 2],
               regularization=[0.0, 0.5, 2],
               frequency_threshold=[0.02, 1.0, 2]
               )
