import processing.dataio.static_data_io as sdio
import processing.processors.static_data_processing as sdpr
import processing.processors.temporal_data_processing as tdpr
import processing.processing_interface as prirf
import pandas
import numpy as np
import processing.parsers as pr
file_path = "/sites/WALLMART/Store0001/Floor1/fingerprints/edited/"
raw_data_df=sdio.load_raw_data(file_path)
processed_data = prirf.process_fingerprints(raw_data_df)
#
# statistics = sdpr.create_statistics_from_raw_data(raw_ble_data_df=raw_data_df)
# features = sdpr.create_features_from_statistics(statistics_df=statistics)
# data = pr.process_features_data(data_features=features)

print ("features are done")

# rolled_features = tdpr.create_features_from_raw_data(raw_data_df,workers=8)
# validation_features = pr.transform_temporal_features(rolled_features,data, full_beacons=raw_data_df['beacon'].unique())
#         # tdpr.transform_temporal_data_to_features()
# print("rolled feature are done")
#
# file_path_validation = "/sites/WALLMART/Store0001/Floor1/fingerprints/validation/"
# raw_data_df_validation=sdio.load_raw_data(file_path_validation)
# statistics_validation = sdpr.create_statistics_from_raw_data(raw_ble_data_df=raw_data_df_validation)
# features_validation = sdpr.create_features_from_statistics(statistics_df=statistics_validation)
# joined_data = pandas.concat([features,features_validation],join='inner',axis=1)
#
# data_1 = joined_data.iloc[:,:features.shape[1]]
# data_2 = joined_data.iloc[:,features.shape[1]:]
# data_1 = data_1.iloc[:,data_1.columns.get_level_values('beacon').isin(data_2.columns.get_level_values('beacon'))]
