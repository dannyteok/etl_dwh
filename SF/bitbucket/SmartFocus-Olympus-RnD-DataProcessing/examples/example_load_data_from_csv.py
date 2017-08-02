import pandas
import processing.io_interface as io
import processing.processing_interface as pr
import ipdb
import shutil
import os
import re

if __name__ == '__main__':
    rootpath = "/Users/panagiotis.agisoikonomou-filandras/Workspace/smartfocus/python_projects/processing/data"

    configuration_json = {
        "method": 'csv',
        "use_max": True,
        "data_path": rootpath,
        "owner": "SMARTFOCUS", "channel": str('0'),
        "split_point": 1.0, "use_mad": False,
        "floor": "Floor8", "site": "London", "use_median": True,
        "features": ['median', 'max', 'frequency'],
        # "filter_beacons": ['2500E7E3000B', '2600E197000B', '2700D0D0000B']
        "filter_beacons": ['2600E3E4000B', '2505A65C000B', '2700E462000B']
    }

    processed_path = os.path.join(rootpath,
                                  configuration_json['owner'],
                                  configuration_json['site'],
                                  configuration_json['floor'],
                                  "fingerprints/processed")

    validation_path = os.path.join(os.path.split(processed_path)[0], 'validation')

    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.mkdir(processed_path)

    if os.path.exists(validation_path):
        shutil.rmtree(validation_path)
    os.mkdir(validation_path)

    pcaComponentsCount = 0
    data = io.load_raw_fingerprint_data(configuration_json)
    full_beacons = set(data[3]['beacon'])
    processed_data = pr.process_static_data(data, pcaComponentsCount=pcaComponentsCount)
    configuration_json['method'] = 'pickle'
    io.save_raw_fingerprint_data(data=data, configuration_json=configuration_json)
    beacons = processed_data['N']['joint'][0].index.get_level_values(level='beacon')
    regexes = [
        re.compile("\^25"),
        re.compile("\^26"),
        re.compile("\^27"),
    ]
    all(any(regex.match(beacon) for regex in regexes) for beacon in beacons)
    combined = "(" + ")|(".join(['^25', '^26', '^27']) + ")"

    window_size = 5
    window_step = 2
    processed_rolling_data = pr.process_rolling_data(data,
                                                     window_size=window_size,
                                                     window_step=window_step,
                                                     features=configuration_json['features']
                                                     )
    io.save_rolled_data(processed_rolling_data, configuration_json=configuration_json)

    validation_data = pr.process_validation_data(processed_rolling_data,
                                                 full_beacons=full_beacons,
                                                 processed_data=processed_data)
    io.save_validation_data(validation_data, configuration_json=configuration_json)
    loaded_rolled_data = io.load_rolled_data(configuration_json=configuration_json)
    validation_data2 = pr.process_validation_data(loaded_rolled_data,
                                                 full_beacons=full_beacons,
                                                 processed_data=processed_data)
