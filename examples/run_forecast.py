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
from converge_load_forecasting import initialise,forecast_pointbased_single_node,forecast_pointbased_multiple_nodes,forecast_inetervalbased_single_node,forecast_inetervalbased_multiple_nodes

# # ==================================================
# # Initialize variables
# # ================================================== 
# # The first step is to create an input_features variable. It can have one of the two following formats.

# # Donwload if data is availbale in csv format
customersdatapath = './NextGen_example.csv'
data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather, input_features = initialise(customersdatapath = customersdatapath,forecasted_param = 'active_power',end_training='2018-12-27',Last_observed_window='2018-12-27',windows_to_be_forecasted=3)

# # Set this value to choose an nmi from customers_nmi 
# # Examples
nmi = customers_nmi_with_pv[10]
n_customers = {i: customers[customers_nmi_with_pv[i]] for i in np.random.default_rng(seed=1).choice(len(customers_nmi_with_pv), size=10, replace=False)}

# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Examples of load forecasting functions
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================

# # ==================================================
# # Method (1): Recursive multi-step point-forecasting method
# # ==================================================

# # generate forecasting values for a specific nmi using a recursive multi-step point-forecasting method
res1 = forecast_pointbased_single_node(customers[nmi],input_features) 
print('forecast_pointbased_single_node is done!')
res1.to_csv('predictions_single_node.csv')

# generate forecasting values for all customers using a recursive multi-step point-forecasting method. 
res2 = forecast_pointbased_multiple_nodes(n_customers,input_features) 
print('forecast_pointbased_multiple_nodes is done!') 
res2.to_csv('predictions.csv')

# # ==================================================
# # Method (2): Recursive multi-step probabilistic forecasting method
# # ==================================================

# generate forecasting values for a specific nmi using a recursive multi-step probabilistic forecasting method
res3 = forecast_inetervalbased_single_node(customers[nmi],input_features)
print('forecast_inetervalbased_single_node is done!')
res3.to_csv('predictions_interval_single_node.csv') 

# generate forecasting values for all customers using a  recursive multi-step probabilistic forecasting method. 
res4 = forecast_inetervalbased_multiple_nodes(n_customers,input_features)
print('forecast_inetervalbased_multiple_nodes is done!')  
res4.to_csv('predictions_interval.csv')


