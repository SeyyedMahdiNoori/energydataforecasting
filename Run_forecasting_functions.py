# Libraries
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
import multiprocessing as mp

core_usage = 1 # 1/core_usage shows core percentage usage we want to use

# # ==================================================================================================
# # Method (1): Recursive multi-step point-forecasting method
# # ==================================================================================================

# Parllel programming approach
# ====================================================================================

# This function is used to parallelised the forecasting for each nmi
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
    
    # Generate predictions 
    customers[customers_nmi].Generate_prediction(input_features)
    
    return customers[customers_nmi].predictions.rename(columns={'pred': customers_nmi}) 

# This function uses the parallelised function and save the result into a single dictionary 
def forecast_pointbased(customers_nmi,input_features):

    predictions_prallel = pool_executor_forecast_pointbased(run_prallel_forecast_pointbased,customers_nmi,input_features)
    
    # Aggregate the results from the parallelised function into a list
    predictions_output = [res for res in predictions_prallel]
    predictions_output = pd.concat(predictions_output, axis=1)

    return predictions_output

# # ==================================================================================================
# # Method (2): Recursive multi-step probabilistic forecasting method
# # ==================================================================================================

# This function is used to parallelised the forecasting for each nmi
def pool_executor_forecast_interval(function_name,customers_nmi,input_features):
    with ProcessPoolExecutor(max_workers=int(cpu_count()/core_usage),mp_context=mp.get_context('fork')) as executor:
        results = executor.map(function_name,customers_nmi,repeat(input_features))  
    return results

# This function outputs the forecasting for each nmi
def run_prallel_Interval_Load_Forecast(customers_nmi,input_features):
    from Load_Forecasting import customers 
    print(" Customer nmi: {first} --------> This is the {second}-th out of {third} customers".format(first = customers_nmi, second=list(customers.keys()).index(customers_nmi),third = len(customers)))


    # Train a forecasting object
    customers[customers_nmi].Generate_forecaster_object(input_features)
    
    # Generate interval predictions 
    customers[customers_nmi].Generate_interval_prediction(input_features)
    
    return customers[customers_nmi].interval_predictions.rename(columns={'pred': customers_nmi})


# This function uses the parallelised function and save the result into a single dictionary 
def forecast_interval(customers_nmi,input_features):
    
    predictions_prallel = pool_executor_forecast_interval(run_prallel_Interval_Load_Forecast,customers_nmi,input_features)

    # Aggregate the results from the parallelised function into a dictionary
    predictions_output_interval = {}
    for res in predictions_prallel:
        predictions_output_interval[int(res.columns[0])] = res.rename(columns={res.columns[0]: 'pred'})
    
    return predictions_output_interval


# Export interval based method into a json file
def export_interval_result_to_json(predictions_output_interval):
    # saving predictions to a json file
    import json
    from copy import deepcopy as copy
    copy_predictions_output = copy(predictions_output_interval)
    for c in copy_predictions_output.keys():
        copy_predictions_output[c] = json.loads(copy_predictions_output[c].to_json())
    with open("my_json_file.json","w") as f:
        json.dump(copy_predictions_output,f)


def read_json_interval():
    
    import json
    import pandas as pd

    with open("my_json_file.json","r") as f:
        loaded_predictions_output = json.load(f)

    for l in list(loaded_predictions_output.keys()):
        loaded_predictions_output[int(l)] = pd.read_json(json.dumps(loaded_predictions_output[l]))
        del loaded_predictions_output[l]
    
    return(loaded_predictions_output)