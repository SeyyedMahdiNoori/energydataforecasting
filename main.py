# ==================================================================================================
# Run to generate the load-forecasting of all the nmis and 
# export it into a csv file
# Two method are implemented:
# 1. Recursive multi-step point-forecasting method
# 2. Recursive multi-step probabilistic forecasting method
# ==================================================================================================


# ==================================================================================================
# Method (1): Recursive multi-step point-forecasting method
# ==================================================================================================


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
# customers_nmi = customers_nmi[0:10]

# # This function is used to parallelised the forecasting for each nmi
# core_usage = 1 # 1/core_usage shows core percentage usage we want to use
# def pool_executor_Backtest(function_name,customers_nmi):
#     with ProcessPoolExecutor(max_workers=int(cpu_count()/core_usage),mp_context=mp.get_context('fork')) as executor:
#         results = executor.map(function_name,customers_nmi)  
#     return results

# # This function outputs the forecasting for each nmi
# def run_prallel_Backtest(customers_nmi):

#     from Load_Forecasting import customers 
#     print(" Customer nmi: {first} --------> This is the {second}-th out of {third} customers".format(first = customers_nmi, second=list(customers.keys()).index(customers_nmi),third = len(customers)))

#     # Run to export the load-forecasting of all nmis in a csv file
#     # ====================================================================================
#     # These functions are used to prevent printing the processing lines of Skforecast librarry 
#     import sys, os
#     # Disable
#     def blockPrint():
#         sys.stdout = open(os.devnull, 'w')
#     # Restore
#     def enablePrint():
#         sys.stdout = sys.__stdout__

#     blockPrint()
#     Temp = customers[customers_nmi].Backtest()[0].rename(columns={'pred': str(customers_nmi)})
#     enablePrint()
#     return Temp 
    

# # Run the parralel function
# predictions_prallel = pool_executor_Backtest(run_prallel_Backtest,customers_nmi)

# # Get the results and generate the csv file
# predictions_output = [res for res in predictions_prallel]
# predictions_output = pd.concat(predictions_output, axis=1)
# predictions_output.to_csv('predictions.csv')



# # ====================================================================================
# # *** Hint ***  To get an nmi's prediction individually use customers[nmi].Backtest()[0]
##
# # Alternative implementions to get the results of "method (1)" using a class method 
# # function build into the customer class and the Python list comprehension method are 
# # as developed below.
# # ====================================================================================

# # Class method function
# # ====================================================================================
# import sys, os
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')
# old_stdout = sys.stdout # backup current stdout
# sys.stdout = open(os.devnull, "w")
# blockPrint()
# from Load_Forecasting import customers_class
# customers_class.Backtest_all()
# predictions_output = pd.concat(customers_class.customer_predictions, axis=1)
# predictions_output.to_csv('predictions.csv')

# # Python list comprehension method
# # ====================================================================================
# import sys, os
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')
# old_stdout = sys.stdout # backup current stdout
# sys.stdout = open(os.devnull, "w")
# blockPrint()
# from Load_Forecasting import customers, customers_nmi
# customer_predictions = [customers[i].Backtest()[0].rename(columns={'pred': str(i)}) for i in customers_nmi]
# predictions_output = pd.concat(customer_predictions, axis=1)
# predictions_output.to_csv('predictions.csv')



# ==================================================================================================
# Method (2): Recursive multi-step probabilistic forecasting method
# ==================================================================================================

# Parllel programming approach
# ====================================================================================
# Libraries
import imp
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
import multiprocessing as mp

# Initialize variables
from Load_Forecasting import customers_nmi

# This function is used to parallelised the forecasting for each nmi
core_usage = 1 # 1/core_usage shows core percentage usage we want to use
def pool_executor_Backtest(function_name,customers_nmi):
    with ProcessPoolExecutor(max_workers=int(cpu_count()/core_usage),mp_context=mp.get_context('fork')) as executor:
        results = executor.map(function_name,customers_nmi)  
    return results

# This function outputs the forecasting for each nmi
def run_prallel_Interval_backtest(customers_nmi):
    from Load_Forecasting import customers 
    print(" Customer nmi: {first} --------> This is the {second}-th out of {third} customers".format(first = customers_nmi, second=list(customers.keys()).index(customers_nmi),third = len(customers)))

    # Run to export the load-forecasting of all nmis in a csv file
    # ====================================================================================
    # These functions are used to prevent printing the processing lines of Skforecast librarry 
    import sys, os
    # Disable
    def blockPrint_par():
        sys.stdout = open(os.devnull, 'w')
    # Restore
    def enablePrint_par():
        sys.stdout = sys.__stdout__

    blockPrint_par()
    Temp = customers[customers_nmi].Interval_backtest()[0].rename(columns={'pred': str(customers_nmi)})
    enablePrint_par()
    return Temp 
    

# Run the parralel function
predictions_prallel = pool_executor_Backtest(run_prallel_Interval_backtest,customers_nmi)

# Get the results and save them into a single dictionary 
predictions_output = {}
for res in predictions_prallel:
    predictions_output[int(res.columns[0])] = res.rename(columns={res.columns[0]: 'pred'})

# saving predictions to a json file
import json
from copy import deepcopy as copy
copy_predictions_output = copy(predictions_output)
for c in copy_predictions_output.keys():
    copy_predictions_output[c] = json.loads(copy_predictions_output[c].to_json())
with open("my_json_file.json","w") as f:
    json.dump(copy_predictions_output,f)


# # To read the predictions from the jason file use the following lines
# with open("my_json_file.json","r") as f:
#     loaded_predictions_output = json.load(f)

# for l in list(loaded_predictions_output.keys()):
#     loaded_predictions_output[int(l)] = pd.read_json(json.dumps(loaded_predictions_output[l]))
#     del loaded_predictions_output[l]


