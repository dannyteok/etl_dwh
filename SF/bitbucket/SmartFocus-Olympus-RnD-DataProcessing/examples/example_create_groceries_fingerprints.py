import processing.io_interface as io_interface
import processing.processing_interface as processing_interface
import joblib
import processing.utils.plots as plots
import processing.utils.helpers as pr
import numpy as np
import copy
from processing.configuration import ProcessingConfiguration

if __name__ == '__main__':
    rootpath = "/sites"

    configuration_json = {
        "method": 'csv',
        "data_path": rootpath,
        "owner": "WALLMART", "channel": str('0'),
        "split_point": 1.0,
        "floor": "Floor1", "site": "Store0001",
        "beacons": ['2700DA1F000B', '2600E312000B', '2700E1F5000B', '260C7941000B', '2500CBA1000B',
                    '250C7739000B', '270C7E2C000B', '2505A65C000B', '2700D474000B', '2500E48F000B',
                    '270C7EB1000B', '2700D0AC000B', '2600E3E4000B', '260C7AB0000B', '250C7A68000B',
                    '2600E4D5000B', '2700E462000B', '250C8159000B', '270C8135000B', '260C7A32000B',
                    '2600DBD7000B', '260C806F000B', '2500CED7000B', '250C7D12000B', '2500E087000B',
                    '270C8195000B', '2600E431000B', '250C7821000B', '2700DE3A000B', '260C7CEF000B',
                    '260C8240000B', '270C814D000B', '260C7BA8000B', '2600D84A000B', '250C7CF0000B',
                    '2500D848000B', '270C7BA0000B', '2600E5AC000B', '2700D920000B', '250C7925000B',
                    '270C7A34000B', '2700D0D0000B', '2600E197000B', '260C790F000B', '2500E7F7000B',
                    '250C7A1D000B', '270C7BAC000B', '2700CC9B000B', '250599AA000B', '2600C32F000B']
    }

    configuration_json['job'] = 0
    configuration_json['pcaComponentsCount'] = 0
    configuration_json['workers'] = 1
    configuration_json['method'] = 'pickle'
    configuration_json['filter_beacons'] = ["2700D0D0000B", "2600E197000B"]
    configuration_json['simulations_count'] = 1
    configuration_json['features'] = ['max', 'median']
    configuration_json['window_size'] = 10
    configuration_json['window_step'] = 1
    configuration_json['filter_data'] = False
    configuration_json['dropObs'] = True
    configuration_json['missing_value'] = -105.0
    configuration_json['plot_type'] = ['rms', 'cdf', 'coordinates', 'heatmap']
    configuration_json['doAndroid'] = True
    configuration_json['observed_value'] = 5.0

    config = ProcessingConfiguration.create_configuration(configuration_json)
    print(config)
    raw_data = io_interface.load_raw_fingerprint_data(configuration=config, suffix='groceries')

    processed_data_groceries = processing_interface.process_fingerprints(
        raw_data,
        configuration=config,
        workers=8,
    )

    raw_data_test = io_interface.load_raw_fingerprint_data(configuration=config, suffix='validation')
    processed_data_groceries_copy = copy.deepcopy(processed_data_groceries)

    processed_data_validation = processing_interface.process_fingerprints(
        raw_data_test,
        configuration=config,
        workers=1,
        processed_data=processed_data_groceries_copy,
    )
    groceries_test_data = {'training': processed_data_groceries, 'validation': processed_data_validation}
    joblib.dump(groceries_test_data, "/sites/WALLMART/Store0001/Floor1/fingerprints/validation/groceries_data.pkl",
                compress=True
                )

    verify_loaded_data = joblib.load("/sites/WALLMART/Store0001/Floor1/fingerprints/validation/groceries_data.pkl")

    training_data = processed_data_groceries['N']['data']
    training_data_coordinates = pr.get_coordinates(training_data)
    plots.plot_rms_error_on_map(training_data_coordinates,
                                rms_error=np.ones(training_data_coordinates.shape[0]),
                                factor=7,
                                image_path="/sites/WALLMART/Store0001/Floor1/image/floor_plan.png"
                                )

    validation_data = processed_data_validation['N']['validation']
    validation_data_coordinates = pr.get_coordinates(validation_data)
    plots.plot_rms_error_on_map(validation_data_coordinates,
                                rms_error=np.ones(validation_data_coordinates.shape[0]),
                                factor=7,
                                image_path="/sites/WALLMART/Store0001/Floor1/image/floor_plan.png"
                                )
