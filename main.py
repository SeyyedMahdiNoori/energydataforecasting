# ==================================================================================================
# Run to generate the load-forecasting of all the nmis and 
# export it into a csv file
# Two method are implemented:
# 1. Recursive multi-step point-forecasting method
# 2. Recursive multi-step probabilistic forecasting method
# *** Hint ***  To get an nmi's prediction individually use 
# customers[nmi].Load_Forecast() or customers[nmi].Interval_Load_Forecast()  
# ==================================================================================================

# ==================================================================================================
# Method (1): Recursive multi-step point-forecasting method
# ==================================================================================================



# Parllel programming approach
# ====================================================================================
# Libraries
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
import multiprocessing as mp

# Initialize variables
from Load_Forecasting import customers_nmi
# To set the data feature like start and end date for training go to the ReadData file
from ReadData import input_features

# This function is used to parallelised the forecasting for each nmi
core_usage = 1 # 1/core_usage shows core percentage usage we want to use
def pool_executor_forecast_pointbased(function_name,customers_nmi,input_features):
    with ProcessPoolExecutor(max_workers=int(cpu_count()/core_usage),mp_context=mp.get_context('fork')) as executor:
        results = executor.map(function_name,customers_nmi,repeat(input_features))  
    return results

# This function outputs the forecasting for each nmi
def run_prallel_forecast_pointbased(customers_nmi,input_features):

    from Load_Forecasting import customers 
    print(" Customer nmi: {first} --------> This is the {second}-th out of {third} customers".format(first = customers_nmi, second=list(customers.keys()).index(customers_nmi),third = len(customers)))

    # Train a forecasting object
    customers[customers_nmi].Generate_forecaster_object(input_features)
    
    # Generate preditions 
    customers[customers_nmi].Generate_prediction(input_features)
    
    return customers[customers_nmi].predictions.rename(columns={'pred': str(customers_nmi)}) 
    

# Run the parralel function
predictions_prallel = pool_executor_forecast_pointbased(run_prallel_forecast_pointbased,customers_nmi,input_features)

# Get the results and generate the csv file
predictions_output = [res for res in predictions_prallel]
predictions_output = pd.concat(predictions_output, axis=1)
predictions_output.to_csv('predictions.csv')

print(predictions_output)

# # ==================================================================================================
# # Method (2): Recursive multi-step probabilistic forecasting method
# # ==================================================================================================

# # Parllel programming approach
# # ====================================================================================
# # Libraries
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor
# from itertools import repeat
# from multiprocessing import cpu_count
# import multiprocessing as mp

# # Initialize variables
# from Load_Forecasting import customers_nmi
# customers_nmi = customers_nmi[0:8]

# # This function is used to parallelised the forecasting for each nmi
# core_usage = 1 # 1/core_usage shows core percentage usage we want to use
# def pool_executor_Backtest(function_name,customers_nmi):
#     with ProcessPoolExecutor(max_workers=int(cpu_count()/core_usage),mp_context=mp.get_context('fork')) as executor:
#         results = executor.map(function_name,customers_nmi)  
#     return results

# # This function outputs the forecasting for each nmi
# def run_prallel_Interval_Load_Forecast(customers_nmi):
#     from Load_Forecasting import customers 
#     print(" Customer nmi: {first} --------> This is the {second}-th out of {third} customers".format(first = customers_nmi, second=list(customers.keys()).index(customers_nmi),third = len(customers)))

#     # Run to export the load-forecasting of all nmis in a csv file
#     # ====================================================================================
#     Temp = customers[customers_nmi].Interval_Load_Forecast().rename(columns={'pred': str(customers_nmi)})
#     return Temp 
    

# # Run the parralel function
# predictions_prallel = pool_executor_Backtest(run_prallel_Interval_Load_Forecast,customers_nmi)

# # Get the results and save them into a single dictionary 
# predictions_output = {}
# for res in predictions_prallel:
#     predictions_output[int(res.columns[0])] = res.rename(columns={res.columns[0]: 'pred'})

# # saving predictions to a json file
# import json
# from copy import deepcopy as copy
# copy_predictions_output = copy(predictions_output)
# for c in copy_predictions_output.keys():
#     copy_predictions_output[c] = json.loads(copy_predictions_output[c].to_json())
# with open("my_json_file.json","w") as f:
#     json.dump(copy_predictions_output,f)


# # # To read the predictions from the jason file use the following lines
# # with open("my_json_file.json","r") as f:
# #     loaded_predictions_output = json.load(f)

# # for l in list(loaded_predictions_output.keys()):
# #     loaded_predictions_output[int(l)] = pd.read_json(json.dumps(loaded_predictions_output[l]))
# #     del loaded_predictions_output[l]


