import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

df = df[df['rssi']<0]

grouped = df.groupby('beacon')

statsBefore = pd.DataFrame({'mean': grouped['rssi'].mean(), 'median': grouped['rssi'].median(), 'std' : grouped['rssi'].std()})

quartile = pd.DataFrame({'q1': grouped['rssi'].quantile(.25), 'median': grouped['rssi'].median(), 'q3' : grouped['rssi'].quantile(.75)})

# Function to detect outlier
def is_outlier(row):
    iq_range = quartile.loc[row.beacon]['q3'] - quartile.loc[row.beacon]['q1']
    median = quartile.loc[row.beacon]['median']
    if row.rssi > (median + (1.5* iq_range)) or row.rssi < (median - (1.5* iq_range)):
        return True
    else:
        return False

# Apply the function to the original df
df.loc[:, 'outlier'] = df.apply(is_outlier, axis = 1)

# Filter to only non-outliers
df_filtered = df[~(df.outlier)]

grouped_filtered = df_filtered.groupby('beacon')

statsAfter = pd.DataFrame({'mean': grouped_filtered['rssi'].mean(), 'median': grouped_filtered['rssi'].median(), 'std' : grouped_filtered['rssi'].std()})

print pd.concat([statsBefore, statsAfter], axis = 1)


df_filtered.drop('outlier', axis = 1, inplace = True)

df_filtered.to_csv("filteredBLE.csv")

key = '270C8234000B'

r1 = (grouped.get_group(key)).rssi

r2 = (grouped_filtered.get_group(key)).rssi

plt.plot(r1, label = 'Before')
plt.plot(r2, label = 'After')
plt.legend(loc='best')
plt.show()
