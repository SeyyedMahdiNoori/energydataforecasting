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

# ###### Pre-process the data ######

# format datetime to pandas datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Add weekday column to the data
data['DayofWeek'] = data['datetime'].dt.day_name()

# Save customer nmis in a list
customers_nmi = list(dict.fromkeys(data['nmi'].values.tolist()))

# #######
# *** Temporary *** the last day of the data (2022-07-31)
# is very different from the rest, and is ommitted for now.
filt = (data['datetime'] < '2022-07-31')
data = data.loc[filt].copy()
# #######

# # Make datetime index of the dataset
data.set_index(['nmi', 'datetime'], inplace=True)

# To obtain the data for each nmi --> data.loc[nmi]



# Set features of the predections
input_features = { 'Forecasted_param': 'active_power',   # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
                    'Start training': '2022-07-01',
                    'End training': '2022-07-27',
                    'Last-observed-window': '2022-07-27',
                    'Window size': 48 ,
                    'Windows to be forecasted':    3      }




# # Check which nmis have PV and their load type
# # ==============================================================================
# # nmi_available = [i for i in customers_nmi if (data_nmi['nmi'] ==  i).any()] # use this line if there are some nmi's in the network that are not available in the nmi.csv file
data_nmi = pd.read_csv('nmi.csv')
data_nmi.set_index(data_nmi['nmi'],inplace=True)
data['has_pv'] = data.active_power.copy()
data['customer_kind'] = data.active_power.copy()
for i in customers_nmi:
    data['has_pv'].loc[i]  = [data_nmi.loc[i]['has_pv']] * len(data['has_pv'].loc[i])
    data['customer_kind'].loc[i]  = [data_nmi.loc[i]['customer_kind']] * len(data['customer_kind'].loc[i])
