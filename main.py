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

# input_features = {  'file_type': 'NextGen',
#                     'file_name': 'LoadPVData.pickle',
#                     'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
#                     'Start training': '2018-01-01',
#                     'End training': '2018-02-01',
#                     'Last-observed-window': '2018-02-01',
#                     'Window size':  288,
#                     'Windows to be forecasted':    3,
#                     'data_freq' : '5T',
#                     'core_usage': 8      }  

# Set features of the predections
input_features = {  'file_type': 'Converge',
                    'file_name': '_WANNIA_8MB_MURESK-nmi-loads.csv',
                    'nmi_type_name': 'nmi.csv',
                    'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
                    'Start training': '2022-07-01',
                    'End training': '2022-07-27',
                    'Last-observed-window': '2022-07-27',
                    'Window size': 48 ,
                    'Windows to be forecasted':    3,     
                    'data_freq' : '30T',
                    'core_usage': 8      
                     }

# Import the required libraries
from load_forecasting_functions import read_data,run_single_forecast_pointbased,forecast_pointbased,run_single_Interval_Load_Forecast,forecast_interval,run_single_demand_disaggregation_optimisation,disaggregation_optimisation,run_single_disaggregate_using_reactive,Generate_disaggregation_using_reactive_all
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


# # ==================================================
# # An example to show some of the functionalities
# # ================================================== 
# # generate forecasting values for a specific nmi using a recursive multi-step point-forecasting method
res1 = run_single_forecast_pointbased(customers[nmi],input_features) 

# generate forecasting values for all customers using a recursive multi-step point-forecasting method. 
res2 = forecast_pointbased(n_customers,input_features)  

 # generate forecasting values for a specific nmi using a recursive multi-step probabilistic forecasting method
res3 = run_single_Interval_Load_Forecast(customers[nmi],input_features)

# generate forecasting values for all customers using a  recursive multi-step probabilistic forecasting method. 
res4 = forecast_interval(n_customers,input_features) 

# disaggregate solar and demand from aggregated data using an optimisation algorithm for a specific time-step
res5 = run_single_demand_disaggregation_optimisation(time_step,customers_nmi_with_pv,datetimes,data_one_time)

# disaggregate solar and demand from aggregated data using an optimisation algorithm for all time-steps
res6 = disaggregation_optimisation(data,input_features,datetimes[0:5],customers_nmi_with_pv)

# disaggregate solar and demand from aggregated data using reactive power as indicator for a specific time-step
res7 = run_single_disaggregate_using_reactive(customers[nmi],input_features)

# disaggregate solar and demand from aggregated data using reactive power as indicator for all time-steps
res8 = Generate_disaggregation_using_reactive_all(n_customers,input_features)







# ==================================================================================================
# Method (1): Recursive multi-step point-forecasting method
# ==================================================================================================

# # Generate forecasting values
# predictions_output = forecast_pointbased(customers,input_features)

# # Export the results into a csv file
# predictions_output.to_csv('predictions.csv')



# ==================================================================================================
# Method (2): Recursive multi-step probabilistic forecasting method
# ==================================================================================================

# # Run the parralel function
# predictions_interval = forecast_interval(customers,input_features)

# # Export the results into a json file
# export_interval_result_to_json(predictions_interval)

# # To read the result from the json file run the following function
# filename = "prediction_interval_based.json"
# loaded_predictions_output = read_json_interval(filename)





