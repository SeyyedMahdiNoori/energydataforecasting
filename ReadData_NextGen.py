import pickle
import copy
import pandas as pd

with open('LoadPVData.pickle', 'rb') as handle:
    data = pickle.load(handle)
data = data[~data.index.duplicated(keep='first')]

datetimes = data.loc[data.index[0][0]].index
customers_nmi = list(data.loc[pd.IndexSlice[:, datetimes[0]], :].index.get_level_values('nmi'))
customers_nmi_with_pv = copy.deepcopy(customers_nmi)


core_usage = 1 # 1/core_usage shows core percentage usage we want to use

# Set features of the predections
input_features = { 'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
                    'Start training': '2018-01-01',
                    'End training': '2018-02-01',
                    'Last-observed-window': '2018-02-01',
                    'Window size':  288,
                    'Windows to be forecasted':    3,
                    'data-freq' : '5T'      }


# To obtain the data for each nmi: --> data.loc[nmi]
# To obtain the data for timestep t: --> data.loc[pd.IndexSlice[:, datetimes[t]], :]

##### Read 5-minute weather data from SolCast
data_weather = pd.read_csv('-35.3075_149.124417_Solcast_PT5M.csv')

data_weather['PeriodStart'] = pd.to_datetime(data_weather['PeriodStart'])
data_weather = data_weather.drop('PeriodEnd', axis=1)
data_weather = data_weather.rename(columns={"PeriodStart": "datetime"})
data_weather.set_index('datetime', inplace=True)
data_weather.index = data_weather.index.tz_convert('Australia/Sydney')

# *** Temporary *** the last day of the data (2022-07-31)
# is very different from the rest, and is ommitted for now.
filt = (data_weather.index > '2018-01-01 23:59:00')
data_weather = data_weather.loc[filt].copy()




## Draw closeness of the global horizontal irradiance to PV generation
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(2)
# axs[0].plot(data_weather.Ghi['2018-03-02'])
# axs[1].plot(data.loc[customers_nmi[90]].pv['2018-03-02'])

# fig, axs = plt.subplots(2)
# axs[0].plot(data_weather.AirTemp['2018-10-02'])
# axs[1].plot(data.loc[customers_nmi[20]].load_active['2018-10-02'])

## Perform a Granger causality test to see which paramters can be used to forecast load and PV
# import numpy as np
# from statsmodels.tsa.stattools import grangercausalitytests
# A = pd.DataFrame(np.transpose(np.array([list(data.loc[customers_nmi[1]].pv['2018-03-02'].values),list(data_weather.Ghi['2018-03-02'].values)])))
# grangercausalitytests(A[[0, 1]], maxlag=[3])

# B = pd.DataFrame(np.transpose(np.array([list(data.loc[customers_nmi[4]].load_active['2018-03-02'].values),list(data_weather.AirTemp['2018-03-02'].values)])))
# grangercausalitytests(B[[0, 1]], maxlag=[3])