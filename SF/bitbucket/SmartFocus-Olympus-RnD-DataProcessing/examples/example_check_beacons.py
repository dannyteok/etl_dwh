import processing.io_interface as io
import processing.processing_interface as pr

if __name__ == '__main__':
    configuration_json = {

        "method": 'csv',
        "use_max": True, "data_path": "/sites", "site": "BELK", "channel": '0',
        "split_point": 1.0, "use_mad": False,
        "floor": "Floor1", "location": "FortWorth", "use_median": True,
        "filter_beacons": ['2500E7E3000B', '2600E197000B', '2700D0D0000B']
    }
    data = io.load_raw_fingerprint_data(configuration_json=configuration_json)

    configuration_json['method'] = 'mongo'


    processed_data = pr.process_fingerprints(data,-1, 5,2)
    # dataio.save_raw_fingerprint_data(data, configuration_json=configuration_json)
