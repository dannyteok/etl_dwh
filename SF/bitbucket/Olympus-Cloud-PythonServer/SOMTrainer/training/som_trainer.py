import sklearn.cluster as skcluster
import scipy.spatial.distance as dist
import numpy as np
import pandas
import ipdb

import som.som_interface as som_interface
import som.soms.tree_som as tree_som
import som.somIO as somio
import processing.io_interface as io_interface
from processing.configuration import ProcessingConfiguration
from som.utils.helpers import get_coordinates

def som_trainer(job_configuration):
    direction = job_configuration['direction']
    iterations = int(job_configuration['iterations'])
    som_type = job_configuration['som_type']
    cluster = int(job_configuration['clusters'])
    max_iterations = int(job_configuration['maxIterations'])
    somSize = int(job_configuration['somSize'])
    channel = job_configuration['channel']
    sigma = float(job_configuration['sigma'])
    learning_rate = float(job_configuration['learning_rate'])
    fuzzy = bool(job_configuration["fuzzy"])
    regions = int(job_configuration["regions"])
    random_permutations = bool(job_configuration['random_permutations'])
    regularization_parameter = float(job_configuration['regularization_parameter'])
    train_expanding = bool(job_configuration['train_expanding'])
    channel = channel[1:] if channel[0] == '^' else channel

    config = ProcessingConfiguration.create_configuration(job_configuration)

    job_configuration['unique_identifier'] = config.identifier
    print("loading processed file with: {}".format(config))
    processed_data = io_interface.load_processed_fingerpint_data(configuration=config)

    if train_expanding:
        training_features = processed_data[direction]['window'].drop_duplicates()
        print("training window features shape: {}".format(training_features.shape))
        validation_features =  processed_data[direction]['validation'].drop_duplicates()
        print("validation features shape: {}".format(validation_features.shape))

    else:
        training_features = processed_data[direction]['data'].drop_duplicates()
        print("training full features shape: {}".format(training_features.shape  ))
        validation_features = processed_data[direction]['validation']
        print("validation features shape: {}".format(validation_features.shape))


    pca_components = processed_data[direction].get('pcaComponentsCount')
    random_seed = job_configuration.get("random_seed",None)

    if som_type == "tree":
        training_coordinates = som_interface.index_to_coordinates(training_features)

        cluster_algorithm = skcluster.KMeans(n_clusters=regions)

        cluster_algorithm.fit(training_coordinates)
        training_labels = cluster_algorithm.predict(training_coordinates)

        som = tree_som.TreeSom(
            feature_size=training_features.shape[1],
            iterations=iterations,
            maximum_iterations=max_iterations,
            sigma=sigma,
            learning_rate=learning_rate,
            fuzzy=fuzzy,
            level_types=["llr", "em", "som"],
            level_sizes=[regions, cluster, somSize],
            random_seed=random_seed,
            regularization_parameter=regularization_parameter,
            random_permutations=random_permutations,
        )

        som.fit(training_features, training_labels)

    else: #{
        print ("som is not a tree som deprecated")
        raise DeprecationWarning
        # som = som_interface.SOMInterface(iterations=iterations, maximum_iterations=max_iterations)
        # som.initialize_som(somSize=somSize, featuresSize=training_features.shape[1], som_type=som_type, cluster=cluster,
        #                    sigma=sigma, learning_rate=learning_rate, fuzzy=fuzzy, random_seed=random_seed
        #                    )
        # som.fit(training_features)


    dictionary = tree_som.get_error_metrics(tree_instance=som,
                                            training_features=training_features,
                                            validation_features=validation_features)
    dictionary['direction'] = direction
    dictionary['channel'] = channel
    dictionary['pca_components'] = pca_components
    dictionary['training_identifier'] = config.identifier
    dictionary['validation_features'] = validation_features
    somio.save_som(dictionary=dictionary, configuration_json=job_configuration)
    return dictionary['error']
