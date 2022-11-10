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
from converge_load_forecasting import initialise,forecast_pointbased_single_node,forecast_pointbased_multiple_nodes,forecast_inetervalbased_single_node,forecast_inetervalbased_multiple_nodes

# # ==================================================
# # Initialize variables
# # ================================================== 
# # The first step is to create an input_features variable. It can have one of the two following formats.


# raw_data read from a server
MURESK_network_data_url = 'https://cloudstor.aarnet.edu.au/sender/download.php?token=087e5222-9919-4c67-af86-3e7d284e1ec2&files_ids=17805910'
raw_data = pd.read_csv(MURESK_network_data_url)
data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather, input_features = initialise(raw_data = raw_data,forecasted_param = 'active_power',end_training='2022-07-12',Last_observed_window='2022-07-12',windows_to_be_forecasted=3)


# # Read if data is availbale in csv format
# customersdatapath = '/Users/mahdinoori/Documents/WorkFiles/Simulations/LoadForecasting/load_forecasting/data/Examples_data/_WANNIA_8MB_MURESK-nmi-loads_example.csv'
# data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather, input_features = initialise(customersdatapath = customersdatapath,forecasted_param ='active_power')


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

# Export the results into a csv file
res2.to_csv('predictions.csv')

# # ==================================================
# # Method (2): Recursive multi-step probabilistic forecasting method
# # ==================================================

# generate forecasting values for a specific nmi using a recursive multi-step probabilistic forecasting method
res3 = forecast_inetervalbased_single_node(customers[nmi],input_features)
print('forecast_inetervalbased_single_node is done!') 

# generate forecasting values for all customers using a  recursive multi-step probabilistic forecasting method. 
res4 = forecast_inetervalbased_multiple_nodes(n_customers,input_features)
print('forecast_inetervalbased_multiple_nodes is done!')  

# Export the results into a json file
from converge_load_forecasting import export_interval_result_to_json
output_file_name = "prediction_interval_based.json"
export_interval_result_to_json(res4,output_file_name)

# # To read the result from the json file run the following function
# from converge_load_forecasting import read_json_interval
# filename = "prediction_interval_based.json"
# loaded_predictions_output = read_json_interval(filename)




