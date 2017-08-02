from processing.dataio import static_data_io as sdio
from processing.dataio import temporal_data_io as tdio
import os



def get_fingerprint_path(configuration_json, suffix=None, method=None):
    path = os.path.join(
        configuration_json.get("data_path"),
        configuration_json.get("owner"),
        configuration_json.get("site"),
        configuration_json.get("floor"),
        "fingerprints")
    if suffix:
        path = os.path.join(path, suffix)

    if method == 'mongo':
        path = os.path.join(
            "/sites",
            configuration_json.get("owner", "SMARTFOCUS"),
            configuration_json.get("site", "London"),
            configuration_json.get("floor", "Floor8"),
            "fingerprints")
    return path


def load_raw_fingerprint_data(configuration, suffix='edited'):
    path = configuration.get_path(suffix=suffix)
    return sdio.load_raw_data(path=path)

def load_raw_validation_data(configuration, suffix, direction, node):
    data = load_raw_fingerprint_data(configuration, suffix)
    data['node'] = node
    data['direction'] = direction
    data = data.sort_values('timestamp')
    return data


def save_raw_fingerprint_data(data, configuration_json):
    method = configuration_json.get("method", 'mongo')
    path = get_fingerprint_path(configuration_json, 'processed', method=method)
    channel = configuration_json["channel"]
    description = configuration_json.get('description', 'Default')
    statistics, analysis, validation, raw_data = data
    sdio.save_raw_fingerprints(path=path,
                               statistics=statistics,
                               analysis=analysis,
                               validation=validation,
                               raw_data=raw_data,
                               channel=channel,
                               method=method,
                               description=description
                               )



def save_processed_fingerpint_data(data, config):
    if data is None:
        return
    description = "Default"
    # method = configuration_json.get("method", 'mongo')
    #
    # path = get_fingerprint_path(configuration_json, 'processed', method)
    # channel = configuration_json["channel"]
    # pcaComponentsCount = configuration_json['pcaComponentsCount']

    path =config.get_path('processed')
    sdio.save_processed_fingerprints(path=path,
                                     pca_data_set=data,
                                     pcaComponentsCount=config.pca_components_count,
                                     identifier=config.identifier,
                                     description=description,
                                     method="pickle",
                                     channel=config.channel
                                     )


def load_rolled_data(configuration_json):
    method = configuration_json.get("method", 'mongo')

    path = get_fingerprint_path(configuration_json, 'processed', method)
    description = configuration_json.get("description", "Default")
    method = configuration_json.get("method", 'mongo')
    channel = configuration_json.get("channel")
    pcaComponentsCount = configuration_json.get('pcaComponentsCount')
    return tdio.load_rolled_data(path=path, description=description, channel=channel,
                                 pcaComponentsCount=pcaComponentsCount, method=method)


def load_processed_fingerpint_data(configuration):
    description = "Default"
    path = configuration.get_path("processed")
    channel = configuration.channel
    n_pcaComponentsCount = configuration.pca_components_count
    unique_identifier = configuration.identifier

    return sdio.load_processed_data(path=path, description=description, channel=channel,
                                    n_pcaComponentsCount=n_pcaComponentsCount, method='pickle',
                                    unique_identifier=unique_identifier)


def save_rolled_data(data, configuration_json):
    method = configuration_json.get("method", 'mongo')

    path = get_fingerprint_path(configuration_json, 'processed', method)
    description = configuration_json.get("description", "Default")
    method = configuration_json.get("method", 'mongo')
    channel = configuration_json.get("channel", '0')

    tdio.save_rolled_data(data, path=path, description=description, method=method, channel=channel)


# def save_validation_data_to_pickle(data)


def save_validation_data(data, configuration_json):
    if data is None:
        return

    method = configuration_json.get("method", 'mongo')

    path = get_fingerprint_path(configuration_json, 'processed', method)
    description = configuration_json.get("description", "Default")
    channel = configuration_json.get("channel", '0')
    pca_pcaComponentsCount = configuration_json.get('pcaComponentsCount', 0)

    sdio.save_processed_fingerprints(path=path,
                                     pca_data_set=data,
                                     pcaComponentsCount=pca_pcaComponentsCount,
                                     description=description,
                                     method=method,
                                     channel=channel,
                                     validation=True
                                     )


def load_validation_data(configuration_json):
    method = configuration_json.get("method", 'mongo')

    path = get_fingerprint_path(configuration_json, 'processed', method)

    description = configuration_json.get("description", "Default")
    method = configuration_json.get("method", 'mongo')
    channel = configuration_json.get("channel", '0')
    n_pcaComponentsCount = configuration_json.get('pcaComponentsCount', 10)
    return sdio.load_validation_data(path=path, description=description, channel=channel,
                                     n_pcaComponentsCount=n_pcaComponentsCount, method=method)
