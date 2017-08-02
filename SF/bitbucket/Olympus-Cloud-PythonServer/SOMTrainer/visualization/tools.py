import SOMTrainer.model.model_interface as model_interface
import processing.io_interface as io_interface
from processing.configuration import ProcessingConfiguration
import som.som_interface as som_interface
import os
import joblib
import SOMTrainer.utils as utils
import numpy as np
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import ipdb
import sklearn.decomposition as dec
import som.soms.tree_som as tree_som



def calculate_rms_per_coordinates(real_coordinates, predicted_coordinates):
    unique_coordinates = utils.unique_rows(real_coordinates)
    rms_error = []
    for coordinates in unique_coordinates:
        indeces =  np.where((real_coordinates == coordinates).all(axis=1))[0]
        coordinate_precitions = predicted_coordinates[indeces,:]
        rms_error.append(np.nanmean(dist.cdist(coordinates[np.newaxis,:],coordinate_precitions)))

    return unique_coordinates, rms_error


def save_accuracy_plot(configuration_json, plot_types=['coordinates']):
    root_path = utils.get_path(configuration_json)
    trained_som_path = os.path.join(root_path, 'fingerprints/somoutput')
    optimal_data_frame_path = os.path.join(root_path, 'fingerprints/validation/som_report.pkl')
    factor = configuration_json['factor']

    if not os.path.exists(optimal_data_frame_path):
        print ("optimal file does not exist")
        return

    validation_data = joblib.load(optimal_data_frame_path)
    optimal_dataframe = validation_data['report']

    image_path = os.path.join(root_path, "image/floor_plan.png")
    direction_list = ['N', 'E', 'S', 'W']
    channel_list = []
    if configuration_json.get("doIOS",True):
        channel_list.append('0')
    if configuration_json.get("doAndroid",True):
        channel_list.extend(['25', '26', '27'])
    for channel in channel_list:
        for som_type in ['tree']:
            for direction in direction_list:
                configuration_json['channel'] = channel
                identity = model_interface.get_som_file_identity(optimal_dataframe,
                                                                 som_type=som_type,
                                                                 channel=channel,
                                                                 direction=direction
                                                                 )
                som_dictionary = model_interface.find_pkl(
                    trained_som_path=trained_som_path,
                    som_type=som_type,
                    channel=channel,
                    direction=direction,
                    identity=identity)
                for plot_type in plot_types:
                    if plot_type == "rms" or plot_type == "coordinates":
                        config = ProcessingConfiguration.create_configuration(configuration_json)
                        processed_data = io_interface.load_processed_fingerpint_data(config)
                        validation_features = processed_data[direction]['validation']
                        plot_rms_error = False
                        if plot_type == "rms":
                            plot_rms_error=True
                        fig = save_predicted_results(validation_features=validation_features,
                                                     som_dictionary=som_dictionary,
                                                     factor=factor,
                                                     image_path=image_path,
                                                     rms=plot_rms_error
                                                     )
                    elif plot_type == "heatmap" or plot_type == "cdf":
                        fig = save_som_plots(som_dictionary=som_dictionary, factor=factor, image_path=image_path,
                                             plot_type=plot_type
                                             )

                    fig.savefig(image_path[:-4] + "-{}-{}-{}-{}.svg".format(plot_type,som_type, channel, direction))
                    plt.close()



def save_predicted_results(validation_features, som_dictionary, factor, image_path,rms=True):

    som = tree_som.TreeSom(
        feature_size=validation_features.shape[1],
        sigma=1,
        learning_rate=0.5,
        iterations=100,
        maximum_iterations=10000,
        level_types=["llr", "em", "som"],
        level_sizes=[1, 1, 10]
    )
    som.set_parameters(som_dictionary)
    # som = som_interface.SOMInterface()
    # som.initialize_som_from_dictionary(som_dictionary)
    predicted_coordinates = som.predict(validation_features)
    real_coordinates = som_interface.index_to_coordinates(validation_features)
    unique_coordinates, rms_error = calculate_rms_per_coordinates(real_coordinates, predicted_coordinates)
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca()
    if rms:
        plots.plot_rms_error_on_map(real_coordinates=unique_coordinates, rms_error=rms_error,
                                    factor=factor,
                                    image_path=image_path,
                                    ax=ax
                                    )

    else:
        plots.plot_coordinates_on_map(real_coordinates=real_coordinates,
                                      predicted_coordinates=predicted_coordinates,
                                      factor=factor,
                                      image_path=image_path,
                                      ax=ax,
                                      )
    return fig


def save_som_plots(som_dictionary, factor, image_path,plot_type='heatmap'):

    wmap = som_dictionary['wmap']
    weights =[]
    real_coordinates = []
    for key, coordinates in wmap.iteritems():
        real_coordinates.append(coordinates[0])
        if len(key)==2:
            weights.append(som_dictionary['weights'][key])
        else:
            weights_som = som_dictionary['layers'][key[0]]
            if weights_som is not None:
                weights_som = weights_som['weights'][key[1],key[2]]
            else:
                weights_som = som_dictionary['cluster']['means'][key[0]]
            weights.append(weights_som)

    weights = np.asarray(weights)
    real_coordinates = np.asarray(real_coordinates)


    def plot_cdf(ax):
        weights_distances= dist.pdist(weights)
        coordinate_distances = dist.pdist(real_coordinates)
        weight_distances_norm = weights_distances/weights_distances.max() * coordinate_distances.max()
        def get_distances_count(weight_value, coordinate_value):

            return np.mean(np.logical_and(weight_distances_norm<=weight_value,
                                          coordinate_distances<=coordinate_value))

        size_t =100
        weight_x = np.linspace(0, weight_distances_norm.max(),size_t)
        coordinates_y = np.linspace(0,coordinate_distances.max(),size_t)
        cdf_surface = np.zeros((size_t,size_t))
        for index, dist_x in enumerate(weight_x):
            for jindex, dist_y in enumerate(coordinates_y):
                cdf_surface[jindex, index] = get_distances_count(dist_x,dist_y)

        img = ax.imshow(cdf_surface)
        plt.colorbar(img, ax=ax)
        ax.set_ylabel("SOM distances scaled")
        ax.set_xlabel("floorplan distances")
        ax.invert_yaxis()


    # def plot_similarity(ax):
    #     pca = dec.PCA(n_components=3)
    #     weights_pca = pca.fit_transform(weights)
    #     range_pca = weights_pca.max(0)-weights_pca.min(0)
    #     weights_pca_scale = (weights_pca + abs(weights_pca.min(0)))/range_pca
    #     plots.plot_similarity_on_map(real_coordinates=real_coordinates,
    #                                  colors=weights_pca_scale,
    #                                  image_path=image_path,
    #                                  ax=ax,
    #                                  factor=factor
    #                                  )
    #
    # fig = plt.figure()
    # ax = plt.gca()
    # if plot_type == 'heatmap':
    #     plot_similarity(ax)
    # else:
    #     plot_cdf(ax)
    # return fig
