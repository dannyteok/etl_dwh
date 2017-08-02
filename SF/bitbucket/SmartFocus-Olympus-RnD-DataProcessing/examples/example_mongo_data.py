import processing.processing_interface as processing_interface
import processing.io_interface as io_interface
import pandas
import processing.constants as constants

if __name__ == '__main__':

    data_path = "/sites/SMARTFOCUS/London/Floor8/fingerprints/edited"
    som_data_path ="/sites/SMARTFOCUS/London/Floor8/fingerprints/som-output/trained-data-som-ios-0-20nodes-6cluster-Falsemad-Truemax-Truemedian-Wdir-10comp-f4adcdf71f534320ae4c220ef7144500.pkl"
    som_data_path2 ="/sites/SMARTFOCUS/London/Floor8/fingerprints/som-output/trained-data-emsom-ios-0-20nodes-6cluster-Falsemad-Truemax-Truemedian-Wdir-10comp-e169b84c76a343eba649127729960069.pkl"

    configuration_json = {

        "method": 'csv',
        "use_max": True, "data_path": "/sites", "site": "SMARTFOCUS", "channel": '0',
        "split_point": 1.0, "use_mad": False,
        "floor": "Floor8", "location": "London", "use_median": True
    }

    for channel in ['0', '25', '26','27']:
        configuration_json['channel'] = channel
        data = io_interface.load_raw_fingerprint_data(configuration_json=configuration_json)

        unique_beacons = pandas.Series.unique(data[-1]['beacon'])
        processed_data = processing_interface.process_static_data(data=data, pcaComponentsCount=10, verbose=False)

        processed_rolling_data = processing_interface.process_rolling_data(data=data,window_size=5, window_step=5)

        validation_features = processing_interface.process_validation_data(rolled_data=processed_rolling_data,
                                                                           processed_data=processed_data,
                                                                           full_beacons=unique_beacons
                                                                           )

        for direction in constants.DIRECTIONS:
            processed_data[direction]['validation'] = validation_features[direction]
        configuration_json['method'] = 'mongo'
        io_interface.save_raw_fingerprint_data(data, configuration_json)
        io_interface.save_rolled_data(processed_rolling_data, configuration_json)
        io_interface.save_processed_fingerpint_data(processed_data, configuration_json)

        # io_interface.load_processed_fingerpint_data()


    # test_data = data_interface.load_validation_data(data_path, 'N')
    # loaded = parsers.process_features_data(data_path, n_pcaComponentsCount=10)

    # statistics, analysis, validation, raw_data = dpr.load_model_fingerprint_data(data_path, full=True)
    # rolling_data = tpr.create_rolled_data(raw_data, window_step=2, window_size=5)
    # # mongoDB.save_csv_data(pathname=data_path, statistics_data=statistics, analysis_data=analysis, raw_data=raw_data, validation_data=validation_data)
    # data_id = mongodb.find_data_set_id(data_path, description='Default')
    #
    #
    #
    # # ipdb.set_trace()
    # mongodb.insert_data(rolling_data, collection_name='BLErolling', data_id=data_id, overwrite=False)
    # #
    # rolled_data_fetched = mongodb.find_dataframe(data_id=data_id, collection_name='BLErolling')
    # rolled_data_fetched = rolled_data_fetched.set_index(
    #     ['node','direction', 'stats','beacon']).sort_index()

    # test_data = parsers.process_data_rolling(data_path, 'E', window_size=5, window_step=2)
        # loaded2 = parsers.process_features_data(path=data_path)

    # features = parsers.load_features(data_path, direction='E', description='Default')

    # som_data = joblib.load('/sites/SMARTFOCUS/London/Floor8/fingerprints/som-output/som_test1.pkl')
    # som_data2 = joblib.load('/sites/SMARTFOCUS/London/Floor8/fingerprints/som-output/som_test2.pkl')
    #
    # data_interface.save_som(som_data2, data_path,'Default')
    # data_interface.save_som(som_data, data_path,'Default')
    #
    # loaded_soms = data_interface.load_soms(path=data_path, direction='N', channel='0')
    #
    # loaded_features = data_interface.load_features(path=data_path, direction='W',channel='0', description='Default')
