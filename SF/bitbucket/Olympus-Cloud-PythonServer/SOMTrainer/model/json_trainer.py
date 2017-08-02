import numpy as np
import pandas
import json
import ipdb
from six import iteritems
import scipy.spatial.distance as dist

# Changes done to move to python 3.5

def sort_dataframe(data, axis=1):
    if axis == 1:
        return data.reindex_axis(sorted(data.columns), axis=1)
    else:
        return data.reindex_axis(sorted(data.index), axis=0)


def reset_json(filename):
    f = open(filename)
    jsonfile = json.load(f)

    for node in jsonfile['graph']['nodes']:
        if len(node['fingerprints']) > 0:
            node['fingerprints'] = []

    f2 = open(filename[:-5] + "-untrained.json", 'w')
    json.dump(jsonfile, f2)
    f2.close()
    f.close()


def key_to_id(key):
    return str(int(key[0])) + "-" + str(int(key[1]))


def get_graph_nodes(value):
    graphnodes = []
    if value and not any(np.isnan(value[0])):
        graphnodes = [key_to_id(identity) for identity in value]
    return graphnodes


def get_weights(weights, index):
    weights = pandas.Series(weights, index=index).unstack(1)
    result = [{'uuid': beacon, 'median': weights[beacon]['median'], 'mad': weights[beacon]['mad']} for beacon in
              weights]
    return result


def get_som_nodes_from_shape(weights_shape):
    som_nodes = []

    for row in np.dstack((np.meshgrid(np.arange(weights_shape[0]),
                                      np.arange(weights_shape[1])))).reshape(-1, 2):
        som_nodes.extend([(row[0], row[1])])
    return list(set(som_nodes))


def rec_get_som_nodes(som, wmap_train, identifier, precision):
    if som.get('cluster'):
        cluster = som['cluster']
        som_weights = cluster['weights']
        som_means = cluster['means']
        node_list = []
        for index in range(cluster['size']):
            sub_layer_nodes = []
            if som.get('layers'):
                if som['layers'][index]:
                    sub_layer_nodes = rec_get_som_nodes(som['layers'][index], wmap_train, identifier=index,
                                                        precision=precision)
            node_list.append({
                'identifier': '{}'.format(index),
                'weight': np.round(som_weights[index], precision).tolist(),
                'means': np.round(som_means[index], precision).tolist(),
                'precision': np.round(np.linalg.pinv(cluster['cov'][index]).flatten(), precision).tolist(),
                'nodes': sub_layer_nodes
            })


    else:
        som_weights = som['weights']
        node_list = []
        som_nodes = get_som_nodes_from_shape((som_weights.shape[0], som_weights.shape[1]))
        for index, node in enumerate(som_nodes):
            if len(wmap_train.keys()[0]) == 2:
                graph_nodes = wmap_train.get(node)
            else:
                graph_nodes = wmap_train.get((identifier, node[0], node[1]))
            graph_nodes_list =  get_graph_nodes(graph_nodes)
            if graph_nodes_list:
                node_list.append({
                    'identifier': key_to_id(node),
                    'weights': np.round(som_weights[node[0], node[1]], precision).tolist(),
                    'graphNodes': graph_nodes_list,
                    'nodes': []
                })

    return node_list


def get_som_nodes(root_node, tree_nodes, identifier, precision):
    wmap_train = root_node['wmap']
    ipdb.set_trace()
    som_weights = root_node['weights']
    node_list = []
    som_nodes = get_som_nodes_from_shape((som_weights.shape[0], som_weights.shape[1]))
    for index, node in enumerate(som_nodes):
        if len(wmap_train.keys()[0]) == 2:
            graph_nodes = wmap_train.get(node)
        else:
            graph_nodes = wmap_train.get((identifier, node[0], node[1]))
        node_list.append({
            'identifier': key_to_id(node),
            'weights': np.round(som_weights[node[0], node[1]], precision).tolist(),
            'graphNodes': get_graph_nodes(graph_nodes),
            'nodes': []
        })
    return node_list


def get_som_nodes_compressed(root_node, tree_nodes, identifier, precision, add_empty_weights):
    som_weights = root_node['weights'].reshape(-1, root_node['weights'].shape[-1])
    node_list = []
    som_size, _, feature_size = root_node['shape']
    for node_key, graph_nodes in iteritems(root_node['wmap']):
        node_weight = som_weights[node_key[0] * som_size + node_key[1]]
        assert node_weight.size == feature_size
        threshold = root_node['tmap'].get(node_key,200)
        if ~any(np.isnan(graph_nodes[0])) or add_empty_weights:
            node = {
                'identifier': key_to_id(node_key),
                'weights': np.round(node_weight, precision).tolist(),
                'graphNodes': get_graph_nodes(graph_nodes),
                'threshold': threshold,
            }
            node_list.append(node)
    return node_list


def get_em_nodes(root_node, tree_nodes, identifier, precision, add_empty_weights, legacy_json):
    em_nodes = []
    for index in range(root_node['shape'][0]):
        som_node = tree_nodes.pop((2, identifier, index))
        cov = root_node['cov'][index]

        if np.all(np.round(root_node['weights'][index], precision) == 1.0 ):
            weight = 1.0
            means = np.round(np.zeros(root_node['shape'][1]),precision).tolist()
        else:
            weight = np.round(root_node['weights'][index], precision).tolist()
            means = np.round(root_node['means'][index],precision).tolist()

        em_node = {'identifier': str(index),
                   'weight': weight,
                   'means': means,
                   'precision': np.round(np.linalg.pinv(cov).flatten(), precision).tolist(),
                   }
        if legacy_json:
            em_node["nodes"]=get_som_nodes_compressed(som_node, tree_nodes, (2, identifier, index), precision,
                                                    add_empty_weights=add_empty_weights)
        else:
            em_node['layer3Nodes']=  get_som_nodes_compressed(som_node, tree_nodes, (2, identifier, index), precision,
                                                        add_empty_weights=add_empty_weights)

        em_nodes.append(em_node)
    return em_nodes


def get_tree_nodes(som, identifier, precision, add_empty_weights, legacy_json=False):
    tree_nodes = som['nodes'].copy()
    root_node = tree_nodes.pop("root")
    node_list = []
    root_categories =som['nodes']['root']['shape'][0]
    for index in range(root_categories):
        em_node = tree_nodes.pop((1, 'root', index))
        if root_categories ==2:
            weights = (2*index-1)*root_node["weights"][0]
            bias = (2*index-1)*root_node["bias"][0]
        else:
            weights = root_node["weights"][index]
            bias = root_node["bias"][index]

        if weights.size != root_node['shape'][1]:
            weights = np.zeros((root_node['shape'][1]))

        node = {
            "bias": np.round(bias, precision).tolist(),
            "weights": np.round(weights, precision).tolist(),
            "identifier": str(index),
        }
        if legacy_json:
            node["nodes"]= get_em_nodes(em_node, tree_nodes, (1, 'root', index), precision,
                                        add_empty_weights=add_empty_weights, legacy_json=legacy_json)
            node_list.append(node)
        else:
            node["layer2Nodes"]= get_em_nodes(em_node, tree_nodes, (1, 'root', index), precision,
                                              add_empty_weights=add_empty_weights, legacy_json=legacy_json)

            node_list.append({"zlayer1Node":node})
    return node_list


def transform_joint_distribution_to_json(joint, channel, direction, precision):
    jpdf_mean = sort_dataframe(joint[0], 0)
    jpdf_cov = sort_dataframe(sort_dataframe(joint[1], 0), 1)
    data_dic = {
        'channel': channel,
        'direction': direction,
        'mean': np.round(jpdf_mean.values.flatten(), precision).tolist(),
        'cov': np.round(jpdf_cov.values.flatten(), precision).tolist(),
        'scope': {
            'stat': jpdf_cov.index.get_level_values('stat').unique().tolist(),
            'beacons': jpdf_cov.index.get_level_values('beacon').unique().tolist()
        }
    }
    return data_dic


def transform_som_dictionary_to_json(som_dictionary, norm, precision, components=None, add_empty_weights=True,
                                     **kwargs):

    legacy_json = kwargs.pop("legacy_json")

    wmap_train = som_dictionary['wmap']

    norm_mean = sort_dataframe(norm[0], 0)
    norm_std = sort_dataframe(norm[1], 0)

    if som_dictionary['type'] == 'tree':
        nodes = get_tree_nodes(som_dictionary, identifier=-1, precision=precision, add_empty_weights=add_empty_weights,
                               legacy_json=legacy_json)
    else:
        nodes = rec_get_som_nodes(som_dictionary, wmap_train, identifier=-1, precision=precision)

    data_dic = {
        "channel": som_dictionary['channel'],
        "direction": som_dictionary["direction"],
        'somType': som_dictionary['type'],
        'normMean': np.round(norm_mean.values.flatten(), precision).tolist(),
        'normStd': np.round(norm_std.values.flatten(), precision).tolist(),
        'scope': {
            "beacons": norm_mean.index.get_level_values("beacon").unique().tolist(),
            "stat": norm_mean.index.get_level_values("stat").unique().tolist(),
        }
    }
    if legacy_json:
        data_dic['nodes']= nodes
    else:
        data_dic['zlayer1Nodes']=nodes


    if components is not None:
        data_dic['pca_components'] = {'values': np.round(components.values.flatten(), precision).tolist(),
                                      'number': components.shape[0]
                                      }
    return data_dic
