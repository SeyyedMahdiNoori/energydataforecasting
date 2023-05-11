# # ==================================================================================================
# # This package is developed as a part of the Converge project at the Australian National University
# # Users can use this package to generate forecasting values for electricity demand. It can also be 
# # used to disaggregate solar and demand from the aggregated measurement at the connection point
# # to the system
# # Two method are implemented to generate forecasting values:
# # 1. Recursive multi-step point-forecasting method
# # 2. Recursive multi-step probabilistic forecasting method
# # For further details on the disaggregation algorithms, please refer to the user manual file
# #
# # Below is an example of the main functionalities in this package
# # ==================================================================================================


from more_itertools import take
import pandas as pd
import numpy as np
from converge_load_forecasting import initialise, forecast_pointbased_multiple_nodes, forecast_inetervalbased_multiple_nodes


# # ==================================================
# # Initialize variables
# # ================================================== 
# # The first step is to create an input_features variable. It can have one of the two following formats.

# # Donwload if data is availbale in csv format
customersdatapath = './NextGen_example.csv'
data_initialised = initialise(customersdatapath = customersdatapath,
                              forecasted_param = 'active_power',
                              end_training='2018-12-29',
                              last_observed_window='2018-12-29',
                              regressor_input = 'LinearRegression',
                              algorithm = 'iterated',
                              loss_function= 'ridge',
                              exog = False,
                              days_to_be_forecasted=1)

# An arbitrary customer nmi to be use as target customer for forecasting
nmi = data_initialised.customers_nmi[10]
customer = {i: data_initialised.customers[nmi] for i in [nmi]}

# n number of customers (here arbitrarily 5 is chosen) to be forecasted parallely
# n_customers = {i: data_initialised.customers[data_initialised.customers_nmi[i]] for i in np.random.default_rng(seed=1).choice(len(data_initialised.customers_nmi), size=5, replace=False)}
n_customers = {i: data_initialised.customers[i] for i in data_initialised.customers_nmi[0:10]}

# n number of customers (here arbitrarily 5 is chosen) with know real-time values
hist_data_proxy_customers = {i: data_initialised.customers[data_initialised.customers_nmi[i]] for i in np.random.default_rng(seed=3).choice(len(data_initialised.customers_nmi), size=5, replace=False) if i not in n_customers.keys()}

# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Examples of load forecasting functions
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================

# # ====================================================================================================
# # Iterated multi-step point-based forecasting method with LinearRegression and Ridge for multiple customersr
# # ====================================================================================================

res_iterated_multiple = forecast_pointbased_multiple_nodes(n_customers,data_initialised.input_features)
res_iterated_multiple.to_csv('res_iterated_multiple.csv')

print('Iterated is done!')

# # ====================================================================================================
# # Direct multi-step point-based forecasting method with LinearRegression, and using Time as exogenous for multiple customers
# # ====================================================================================================

data_initialised.input_features['algorithm'] = 'direct'
res_direct_multiple = forecast_pointbased_multiple_nodes(n_customers,data_initialised.input_features)
res_direct_multiple.to_csv('res_direct_multiple.csv')

print('Direct is done!')

# # ====================================================================================================
# # Stacking multi-step point-based forecasting method with LinearRegression, and using Time as exogenous for multiple customers
# # ====================================================================================================

data_initialised.input_features['algorithm'] = 'stacking'
res_stacking_multiple = forecast_pointbased_multiple_nodes(n_customers,data_initialised.input_features)
res_stacking_multiple.to_csv('res_stacking_multiple.csv')

print('Stacking is done!')

# # ====================================================================================================
# # Rectified multi-step point-based forecasting method with LinearRegression, and using Time as exogenous for multiple customers
# # ====================================================================================================

data_initialised.input_features['algorithm'] = 'rectified'
res_rectified_multiple = forecast_pointbased_multiple_nodes(n_customers,data_initialised.input_features)
res_rectified_multiple.to_csv('res_rectified_multiple.csv')

print('Rectified is done!')

# # ==================================================
# # Recursive multi-step probabilistic forecasting method
# # ==================================================

# generate forecasting values for a specific nmi using a recursive multi-step probabilistic forecasting method
data_initialised.input_features['algorithm'] = 'iterated'
res_interval_multiple = forecast_inetervalbased_multiple_nodes(n_customers,data_initialised.input_features)
res_interval_multiple.to_csv('res_interval_multiple.csv')

print('interval-based is done!')


