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

# # ==================================================
# # Initialize variables
# # ================================================== 
# # The first step is to create an input_features variable. It can have one of the two following formats.

# Set features of the predections
input_features = {  'file_type': 'NextGen',
                    'file_name': 'NextGen.csv',
                    'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
                    'Start training': '2018-01-01',
                    'End training': '2018-02-01',
                    'Last-observed-window': '2018-02-01',
                    'Window size':  288,
                    'Windows to be forecasted':    3,
                    'data_freq' : '5T',
                    'core_usage': 8      }  

# # Set features of the predections
# input_features = {  'file_type': 'Converge',
#                     'data_path':  '/Users/mahdinoori/Documents/WorkFiles/Simulations/LoadForecasting/load_forecasting/data/_WANNIA_8MB_MURESK-nmi-loads.csv',
#                     'nmi_type_path': '/Users/mahdinoori/Documents/WorkFiles/Simulations/LoadForecasting/load_forecasting/data/nmi.csv',
#                     'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
#                     'Start training': '2022-07-01',
#                     'End training': '2022-07-27',
#                     'Last-observed-window': '2022-07-27',
#                     'Window size': 48 ,
#                     'Windows to be forecasted':    3,     
#                     'data_freq' : '30T',
#                     'core_usage': 8      
#                      }

# Import the required libraries
from converge_load_forecasting import read_data,forecast_pointbased_single_node,forecast_pointbased_multiple_nodes,forecast_inetervalbased_single_node,forecast_inetervalbased_multiple_nodes
from more_itertools import take
import pandas as pd

# Read data 
data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather = read_data(input_features)
# To obtain the data for each nmi: --> data.loc[nmi]
# To obtain the data for timestep t: --> data.loc[pd.IndexSlice[:, datetimes[t]], :]



# some arbitarary parameters
n_customers = dict(take(4, customers.items()))     # take n customers from all the customer (to speed up the calculations)
time_step = 144
nmi = customers_nmi_with_pv[0] # an arbitary customer nmi
data_one_time = data.loc[pd.IndexSlice[:, datetimes[time_step]], :]   # data for a time step "time_step" 


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

# generate forecasting values for all customers using a recursive multi-step point-forecasting method. 
res2 = forecast_pointbased_multiple_nodes(n_customers,input_features) 
print('forecast_pointbased_multiple_nodes is done!') 

# # Export the results into a csv file
# res2.to_csv('predictions.csv')

# # ==================================================
# # Method (2): Recursive multi-step probabilistic forecasting method
# # ==================================================

# generate forecasting values for a specific nmi using a recursive multi-step probabilistic forecasting method
res3 = forecast_inetervalbased_single_node(customers[nmi],input_features)
print('forecast_inetervalbased_single_node is done!') 

# generate forecasting values for all customers using a  recursive multi-step probabilistic forecasting method. 
res4 = forecast_inetervalbased_multiple_nodes(n_customers,input_features)
print('forecast_inetervalbased_multiple_nodes is done!')  

# # Export the results into a json file
# from converge_load_forecasting import export_interval_result_to_json
# export_interval_result_to_json(res4)

# # To read the result from the json file run the following function
# from converge_load_forecasting import read_json_interval
# filename = "prediction_interval_based.json"
# loaded_predictions_output = read_json_interval(filename)



# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Examples of solar demand disaggregation functions
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================



