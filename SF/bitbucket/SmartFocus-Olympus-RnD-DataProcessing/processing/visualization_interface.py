import numpy as np
import matplotlib.pyplot as plt
import utils.plots as plots
import os
from processing.configuration import ProcessingConfiguration
import processing.io_interface as io_interface
from processing.utils.helpers import  get_coordinates

def plot_observation_count(statistics_df,factor, image_path, resolution=100):
    coordinates_str = statistics_df.index.get_level_values('node').unique()
    max_observations = np.zeros(len(coordinates_str))
    statistics_coordinates = np.zeros((len(coordinates_str), 2))
    for index, coordinates in enumerate(coordinates_str):
        max_observations[index] = statistics_df.xs(coordinates, level='node').unstack('beacon').fillna(0)['obs'].sum(
            axis=1).max()
        statistics_coordinates[index, 0] = float(coordinates.split('-')[0])
        statistics_coordinates[index, 1] = float(coordinates.split('-')[1])

    fig, ax = plt.subplots()
    plots.plot_observation_heatmap(statistics_coordinates,
                                   max_observations,
                                   factor=factor,
                                   image_path=image_path,
                                   ax=ax,
                                   resolution=resolution
                                   )
    plt.savefig(os.path.join(os.path.split(image_path)[0], "fingerprint_heatmap.png"))


def plot_finger_prints(configuration_json):

    config = ProcessingConfiguration.create_configuration(configuration_json)
    raw_data = io_interface.load_raw_fingerprint_data(config)
    true_coordinates = get_coordinates(raw_data)
    fig, ax = plt.subplots()

    plots.plot_coordinates_on_map(true_coordinates,true_coordinates,factor=configuration_json['factor'],
                                  image_path=config.image_path,
                                  ax=ax,
                                  )
    plt.savefig(os.path.join(os.path.split(config.image_path)[0], "fingerprints.png"))
