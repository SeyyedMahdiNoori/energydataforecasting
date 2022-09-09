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
input_features = {'Start training': '2022-07-01',
                     'End training': '2022-07-27',
                     'Last-observed-window': '2022-07-27',
                     'Window size': 48 ,
                     'Windows to be forecasted':    3      }