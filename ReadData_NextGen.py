import pickle
import copy
import pandas as pd

with open('LoadPVData.pickle', 'rb') as handle:
    data = pickle.load(handle)


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



