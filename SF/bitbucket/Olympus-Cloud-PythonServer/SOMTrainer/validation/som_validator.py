import pandas
import numpy as np
import scipy.spatial.distance as dist
import ipdb
import som.somIO as somio
from collections import namedtuple
import copy


def calculate_RMSError(real_positions: np.ndarray, predicted_positions: np.ndarray) -> float:
    error = np.diag(dist.cdist(real_positions, predicted_positions))
    return np.sqrt(np.nanmean(error ** 2))


def som_aggregator(loaded_soms: list, job_configuration: dict) -> dict:
    soms_per_region_count = {}

    for som_dic in loaded_soms:
        features = ",".join(job_configuration.get('features', ['median', 'max']))

        dictionary = {'window_size': job_configuration['window_size'],
                      'features': features
                      }
        dictionary.update(som_dic.copy())

        soms_per_region_count.setdefault(som_dic['level_sizes'][0], []).append(dictionary)

    return soms_per_region_count


def set_similarity(set_a: set, set_b: set) -> float:
    return len(set_a & set_b) / len(set_a | set_b)


def update_optimal_region(region_a: namedtuple, region_b: namedtuple) -> bool:
    if set_similarity(region_a.region_nodes, region_b.region_nodes) > 0.9:
        if region_a.rmse > region_b.rmse: return True
    return False


def compare_regions(optimal_som_dic: dict, som_dic: dict) -> dict:
    optimal_som_has_changed = False
    if optimal_som_dic['level_sizes'][0]>1:
        for key_a, region_a in optimal_som_dic['regional_error'].items():
            for key_b, region_b in som_dic['regional_error'].items():
                if update_optimal_region(region_a, region_b):
                    optimal_som_dic['nodes'][key_a] = som_dic['nodes'][key_b]
                    optimal_som_dic['regional_error'][key_a] = som_dic['regional_error'][key_b]
                    optimal_som_dic['wmap'][key_a] = som_dic['wmap'][key_b]
                    print ("changed: {0} with {1} from {2} with error {3}".format(key_a, key_b, som_dic['id'],region_b.rmse))
                    optimal_som_has_changed = True
                    break

        weights = [region.positions for region in optimal_som_dic['regional_error'].values()]

        optimal_som_dic['error'] = np.average(
            [region.rmse for region in optimal_som_dic['regional_error'].values()],
            weights=weights
        )
        optimal_som_dic['error_quant'] = np.average(
            [region.quantile for region in optimal_som_dic['regional_error'].values()],
            weights=weights)

        optimal_som_dic['error_std'] = np.average(
            [region.std for region in optimal_som_dic['regional_error'].values()],
            weights=weights)

        optimal_som_dic['nans'] = np.average(
            [region.nans for region in optimal_som_dic['regional_error'].values()],
            weights=weights)
    else:
        if optimal_som_dic['error'] < som_dic['error']:
            optimal_som_dic = som_dic
            optimal_som_has_changed = True

    optimal_som_dic['changed'] = optimal_som_has_changed or optimal_som_dic.get('changed', False)


    return optimal_som_dic


def error_minimizer_per_region():
    optimal_som_dic = None
    while True:
        som_dic = yield
        if som_dic is None:
            break
        optimal_som_dic = som_dic if optimal_som_dic is None else compare_regions(optimal_som_dic, som_dic)

    return optimal_som_dic


def grouper(results, key):
    while True:
        results[key] = yield from error_minimizer_per_region()


def som_validator(job_configuration: dict):
    som_folder = job_configuration.get('som_folder','somoutput')
    loaded_soms = somio.load_soms(job_configuration, suffix=som_folder)
    soms_by_region = som_aggregator(loaded_soms, job_configuration)
    results = {}
    for key, values in soms_by_region.items():
        group = grouper(results, key)
        next(group)
        for value in values:
            group.send(value)
        group.send(None)

    edited_results = {}
    for key, dictionary in results.items():
        som_shape = tuple(dictionary['level_sizes']) if dictionary['type'] == 'tree' else 1
        if dictionary['changed']: dictionary['id'] = somio.save_som(dictionary=dictionary,
                                                                    configuration_json=job_configuration)

        edited_results[key] = {'direction': job_configuration['direction'],
                               'channel': job_configuration['channel'],
                               'id': str(dictionary["id"]),
                               'error': dictionary['error'],
                               'quant': dictionary.get('error_quant'),
                               'std': dictionary.get('error_std'),
                               'pcaComponentsCount': job_configuration['pcaComponentsCount'],
                               'som_type': job_configuration['som_type'],
                               'som_shape': som_shape,
                               'nans': dictionary.get('nans'),
                               'window_size': job_configuration['window_size'],
                               'features': dictionary['features'],
                               }

    return edited_results


def results_generator(results):
    results_copy = copy.copy(results)
    if isinstance(results_copy[0], dict):
        for result in results_copy:
            yield from result.values()
    else:
        yield from results_copy


def create_som_report(results):
    final_results_df = pandas.DataFrame([result for result in results_generator(results)])
    final_results_df = final_results_df[final_results_df['error'] >= 0]
    grouped = final_results_df.groupby(
        ['direction', 'channel', 'som_type', 'som_shape', 'features', 'pcaComponentsCount', 'window_size'])
    report = grouped.min()
    rows = [value for value in final_results_df if value in ['error', 'id', 'nans', 'quant', 'std']]
    report.columns = pandas.Index(rows)
    return report
