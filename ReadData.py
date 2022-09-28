import pandas as pd
# import numpy as np


# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


# ###### Read data ######
# ==============================================================================
# Read data
data = pd.read_csv('_WANNIA_8MB_MURESK-nmi-loads.csv')

# # ###### Pre-process the data ######

# format datetime to pandas datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Add weekday column to the data
data['DayofWeek'] = data['datetime'].dt.day_name()

# Save customer nmis in a list
customers_nmi = list(dict.fromkeys(data['nmi'].values.tolist()))

# *** Temporary *** the last day of the data (2022-07-31)
# is very different from the rest, and is ommitted for now.
filt = (data['datetime'] < '2022-07-31')
data = data.loc[filt].copy()

# Make datetime index of the dataset
data.set_index(['nmi', 'datetime'], inplace=True)

# save unique dates of the data
datetimes = data.index.unique('datetime')

# To obtain the data for each nmi: --> data.loc[nmi]
# To obtain the data for timestep t: --> data.loc[pd.IndexSlice[:, datetimes[t]], :]


core_usage = 1 # 1/core_usage shows core percentage usage we want to use

# Set features of the predections
input_features = { 'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
                    'Start training': '2022-07-01',
                    'End training': '2022-07-27',
                    'Last-observed-window': '2022-07-27',
                    'Window size': 48 ,
                    'Windows to be forecasted':    3      }


# Add PV instalation and size, and load type to the data from nmi.csv file
# ==============================================================================
# nmi_available = [i for i in customers_nmi if (data_nmi['nmi'] ==  i).any()] # use this line if there are some nmi's in the network that are not available in the nmi.csv file
data_nmi = pd.read_csv('nmi.csv')
data_nmi.set_index(data_nmi['nmi'],inplace=True)

import itertools
customers_nmi_with_pv = [ data_nmi.loc[i]['nmi'] for i in customers_nmi if data_nmi.loc[i]['has_pv']==True ]
data['has_pv']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['has_pv']] for i in customers_nmi]* len(datetimes)))
data['customer_kind']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['customer_kind']] for i in customers_nmi]* len(datetimes)))
data['pv_system_size']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['pv_system_size']] for i in customers_nmi]* len(datetimes)))

# # This is line is added to prevent the aggregated demand from being negative when there is not PV
# # Also, from the data, it seems that negative sign is a mistake and the positive values make more sense in those nmis
# # for i in customers_nmi:
# #     if data.loc[i].pv_system_size[0] == 0:
# #         data.at[i,'active_power'] = data.loc[i].active_power.abs()