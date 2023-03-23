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
from converge_load_forecasting import initialise,forecast_pointbased_autoregressive_single_node,forecast_pointbased_autoregressive_multiple_nodes,forecast_inetervalbased_multiple_nodes,forecast_inetervalbased_single_node,forecast_pointbased_rectified_single_node,forecast_pointbased_rectified_multiple_nodes,forecast_pointbased_direct_single_node,forecast_pointbased_direct_multiple_nodes,forecast_pointbased_stacking_single_node,forecast_lin_reg_proxy_measures_single_node,forecast_lin_reg_proxy_measures_separate_time_steps,forecast_pointbased_exog_reposit_single_node

# # ==================================================
# # Initialize variables
# # ================================================== 
# # The first step is to create an input_features variable. It can have one of the two following formats.

# # Donwload if data is availbale in csv format
customersdatapath = './NextGen_example.csv'
data_initialised = initialise(customersdatapath = customersdatapath,forecasted_param = 'active_power',end_training='2018-12-29',Last_observed_window='2018-12-29',windows_to_be_forecasted=1)

# An arbitrary customer nmi to be use as target customer for forecasting
nmi = data_initialised.customers_nmi[10]
customer = data_initialised.customers[nmi]

# n number of customers (here arbitrarily 5 is chosen) to be forecasted parallely
n_customers = {i: data_initialised.customers[data_initialised.customers_nmi[i]] for i in np.random.default_rng(seed=1).choice(len(data_initialised.customers_nmi), size=5, replace=False)}

# n number of customers (here arbitrarily 5 is chosen) with know real-time values
hist_data_proxy_customers = {i: data_initialised.customers[data_initialised.customers_nmi[i]] for i in np.random.default_rng(seed=3).choice(len(data_initialised.customers_nmi), size=5, replace=False) if i not in n_customers.keys()}

# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Examples of load forecasting functions
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================

# # ==================================================
# # Recursive autoregressive multi-step point-forecasting method
# # ==================================================

# # generate forecasting values for a specific nmi using a recursive multi-step point-forecasting method
res_autoregressive_single = forecast_pointbased_autoregressive_single_node(customer,data_initialised.input_features)
res_autoregressive_single.to_csv('res_autoregressive_single.csv')
# res_autoregressive_multi = forecast_pointbased_autoregressive_multiple_nodes(n_customers,data_initialised.input_features)      # # generate forecasting values for selected customers using a recursive multi-step point-forecasting method. 

print('autoregressive is done!')

# # ==================================================
# # Recursive multi-step probabilistic forecasting method
# # ==================================================

# generate forecasting values for a specific nmi using a recursive multi-step probabilistic forecasting method
res_interval_single = forecast_inetervalbased_single_node(customer,data_initialised.input_features)
res_interval_single.to_csv('res_interval_single.csv')
# res_interval_multi = forecast_inetervalbased_multiple_nodes(n_customers,data_initialised.input_features)

print('interval-based is done!')

# ================================================================
# Direct recursive multi-step point-forecasting method
# ================================================================
res_direct_single = forecast_pointbased_direct_single_node(customer,data_initialised.input_features)
res_direct_single.to_csv('res_direct_single.csv')
# res_direct_multi = forecast_pointbased_direct_multiple_nodes(n_customers,data_initialised.input_features)

print('direct is done!')

# # ================================================================
# # Stacking recursive multi-step point-forecasting method
# # ================================================================
res_stacking_single = forecast_pointbased_stacking_single_node(customer,data_initialised.input_features)
res_stacking_single.to_csv('res_stacking_single.csv')
# res_stacking_multi = forecast_pointbased_stacking_multiple_nodes(n_customers,data_initialised.input_features)

print('stacking is done!')


# # ================================================================
# # Recitifed recursive multi-step point-forecasting method
# # ================================================================
res_rectified_single = forecast_pointbased_rectified_single_node(customer,data_initialised.input_features)
res_rectified_single.to_csv('res_rectified_single.csv')
# res_rectified_multi = forecast_pointbased_rectified_multiple_nodes(n_customers,data_initialised.input_features)

print('recitifed is done!')

# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meters
# # ================================================================
res_rep_lin_single_time_single = forecast_lin_reg_proxy_measures_single_node(hist_data_proxy_customers,customer,data_initialised.input_features)
res_rep_lin_single_time_single.to_csv('res_rep_lin_single_time_single.csv')

print('Load_forecasting Using linear regression of Reposit data and smart meters is done!')

# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter, one for each time-step in a day
# # ================================================================
res_rep_lin_multi_time_single = forecast_lin_reg_proxy_measures_separate_time_steps(hist_data_proxy_customers,customer,data_initialised.input_features)    
res_rep_lin_multi_time_single.to_csv('res_rep_lin_multi_time_single.csv')

print('Load_forecasting Using linear regression of Reposit data and smart meter, one for each time-step in a day is done!')

# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter
# # ================================================================
res_rep_exog = forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers,customer,data_initialised.input_features)
res_rep_exog.to_csv('res_rep_exog.csv')

print('Load_forecasting Using linear regression of Reposit data and smart meter is done!')

