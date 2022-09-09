# ==================================================================================================
# Run to generate the load-forecasting of all the nmis and 
# export it into a file
# Two method are implemented:
# 1. Recursive multi-step point-forecasting method
# 2. Recursive multi-step probabilistic forecasting method
# *** Hint ***  To get an nmi's prediction individually use 
# customers[nmi].Load_Forecast() or customers[nmi].Interval_Load_Forecast()  
# ==================================================================================================

# Initialize variables
from Load_Forecasting import customers_nmi
# customers_nmi = customers_nmi[0:8]  # Use this line for testing (the number of nmi in the _WANNIA_8MB_MURESK-nmi-loads.csv is 1292. This line takes the first 8 and produces the results)
from ReadData import input_features # To set the data feature like start and end date for training go to the ReadData file


# ==================================================================================================
# Method (1): Recursive multi-step point-forecasting method
# ==================================================================================================

# import forecasting functions
from Run_forecasting_functions import forecast_pointbased

# Generate forecasting values
predictions_output = forecast_pointbased(customers_nmi,input_features)

# Export the results into a csv file
predictions_output.to_csv('predictions.csv')



# ==================================================================================================
# Method (2): Recursive multi-step probabilistic forecasting method
# ==================================================================================================

# import forecasting functions
from Run_forecasting_functions import forecast_interval, export_interval_result_to_json

# Run the parralel function
predictions_interval = forecast_interval(customers_nmi,input_features)

# Export the results into a json file
export_interval_result_to_json(predictions_interval)

# # To read the result from the json file run the following function
# from Run_forecasting_functions import read_json_interval
# loaded_predictions_output = read_json_interval()





