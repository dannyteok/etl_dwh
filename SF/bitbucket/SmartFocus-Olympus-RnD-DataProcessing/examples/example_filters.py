from __future__ import division, print_function

import numpy as np
import pandas

import processing.parsers as pr
import processing.processors.temporal_data_processing as tp
import processing.utils.filters as filters


if __name__ == '__main__':
    path ="/Projects/Smartfocus-Olympus/OlympusData/db/sites/BELK/Charlotte/Floor1/Fingerprints/edited/set0/"
    file_name = "Floor 01-Bluetooth.csv"
    raw_data = pr.load_csv_fingerprint(path, file_name)
    filtered_sagvol = filters.sagvol_filter(raw_data)
    filtered_quantile = filters.quantile_filter(raw_data)
    filtered_butter = filters.butter_filter(raw_data)
    filtered_ar = filters.ar_filter(raw_data)



    raw_grouped = raw_data.set_index('timestamp').groupby(['node', 'direction', 'beacon'])
    final_data = {}
    for key, group in raw_grouped:
        raw_rssi = group['rssi']
        raw_rssi.name= 'raw'
        sagvol_rssi = filtered_sagvol.xs(key, level=['node', 'direction', 'beacon'])['rssi']
        sagvol_rssi.name='sagvol'
        quantile_Rssi = filtered_quantile.xs(key, level=['node', 'direction', 'beacon'])[
            'rssi']
        quantile_Rssi.name='quant'
        butter_rssi = filtered_butter.xs(key, level=['node', 'direction', 'beacon'])['rssi']
        butter_rssi.name = 'butter'
        ar_rssi = filtered_ar.xs(key, level=['node', 'direction', 'beacon'])['rssi']
        ar_rssi.name = 'AR'
        final_data[key] = pandas.concat([raw_rssi, sagvol_rssi, quantile_Rssi, butter_rssi, ar_rssi], axis=1)

    # box plot example
    final_data.items()[55][1].boxplot(whis=1.5)
    test_data = final_data.items()[55][1]
    test_data = tp.normalize_time(test_data)
    timestamp = 0.01
    Fs = 1./timestamp
    t = np.arange(0, 1, timestamp)
    # test_data_sampled = test_data.resample('10ms').interpolate()
    # freqz = np.fft.fftfreq(test_data_sampled.shape[0], d=timestamp)
    # signal = np.fft.fft(test_data_sampled.values)

# from numpy import sin, linspace, pi
# from pylab import plot, show, title, xlabel, ylabel, subplot
# from scipy import fft, arange
#
#
#
# Fs = 150.0;  # sampling rate
# Ts = 1.0/Fs; # sampling interval
# t = arange(0,1,Ts) # time vector
#
# ff = 5;   # frequency of the signal
# y = sin(2*pi*ff*t)
#
# subplot(2,1,1)
# plot(t,y)
# xlabel('Time')
# ylabel('Amplitude')
# subplot(2,1,2)
# plotSpectrum(y,Fs)
# show()
