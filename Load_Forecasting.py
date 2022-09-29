# Libraries
# ==============================================================================
# General
import imp
import pandas as pd
import numpy as np

# Modelling and Forecasting
from sklearn.linear_model import Ridge # Linear least squares with l2 regularization. (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
from sklearn.pipeline import make_pipeline # Construct a Pipeline from the given estimators (for more information: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance.
from skforecast.ForecasterAutoreg import ForecasterAutoreg # A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
from skforecast.model_selection import grid_search_forecaster, backtesting_forecaster
from datetime import date, timedelta

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
import multiprocessing as mp

from pyomo.environ import NonNegativeReals, ConstraintList, ConcreteModel, Var, Objective
from pyomo.opt import SolverFactory

from tqdm import tqdm
from functools import partialmethod

import json
from copy import deepcopy as copy

# Get data from the ReadData script
from ReadData_NextGen import data, customers_nmi, input_features, datetimes, customers_nmi_with_pv, core_usage

# customers_nmi = customers_nmi_with_pv[0:2]
# customers_nmi_with_pv = customers_nmi_with_pv[0:2]



# Warnings configuration
import warnings
warnings.filterwarnings('ignore')


# Define a class for all the nmis and load forecating functions
# ==============================================================================
class customers_class:
  

    def __init__(self, nmi,input_features):

        self.nmi = nmi      # store nmi in each object              
        self.data = data.loc[self.nmi]      # store data in each object         
        # self.data_train = self.data.loc[input_features['Start training']:input_features['End training']]

    def Generate_forecaster_object(self,input_features):
        
        """
        Generate_forecaster_object(self,input_features)
        
        This function generates a forecaster object to be used for a recursive multi-step forecasting method. 
        It is based on a linear least squares with l2 regularization method. Alternatively, LinearRegression() and Lasso() that
        have different objective can be used with the same parameters.
        
        input_features is a dictionary. To find an example of its format refer to the ReadData.py file
        """

        # Create a forecasting object
        self.forecaster = ForecasterAutoreg(
                regressor = make_pipeline(StandardScaler(), Ridge()),  
                lags      = input_features['Window size']      # The used data set has a 30-minute resolution. So, 48 denotes one full day window
            )

        # Train the forecaster using the train data
        self.forecaster.fit(y=self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']])

    def Generate_optimised_forecaster_object(self,input_features):
        
        """
        Generate_optimised_forecaster_object(self,input_features)
        
        This function generates a forecaster object for each \textit{nmi} to be used for a recursive multi-step forecasting method.
        It builds on function Generate\_forecaster\_object by combining grid search strategy with backtesting to identify the combination of lags 
        and hyperparameters that achieve the best prediction performance. As default, it is based on a linear least squares with \textit{l2} regularisation method. 
        Alternatively, it can use LinearRegression() and Lasso() methods to generate the forecaster object.

        input_features is a dictionary. To find an example of its format refer to the ReadData.py file
        """

        # This line is used to hide the bar in the optimisation process
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        self.forecaster = ForecasterAutoreg(
                regressor = make_pipeline(StandardScaler(), Ridge()),
                lags      = input_features['Window size']      # The used data set has a 30-minute resolution. So, 48 denotes one full day window
            )

        # Regressor's hyperparameters
        param_grid = {'ridge__alpha': np.logspace(-3, 5, 10)}
        # Lags used as predictors
        lags_grid = [list(range(1,24)), list(range(1,48)), list(range(1,72)), list(range(1,96))]

        # optimise the forecaster
        grid_search_forecaster(
                        forecaster  = self.forecaster,
                        y           = self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']],
                        param_grid  = param_grid,
                        # lags_grid   = lags_grid,
                        steps       =  48, # input_features['Window size'],
                        metric      = 'mean_absolute_error',
                        # refit       = False,
                        initial_train_size = len(self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']]) - input_features['Window size'] * 10,
                        # fixed_train_size   = False,
                        return_best = True,
                        verbose     = False
                 )
        

    def Generate_prediction(self,input_features):
        """
        Generate_prediction(self,input_features)
        
        This function outputs the prediction values using a Recursive multi-step point-forecasting method. 
        
        input_features is a dictionary. To find an example of its format refer to the ReadData.py file
        """
        
        Newindex = pd.date_range(start=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=1), end=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=input_features['Windows to be forecasted']+1),freq=input_features['data-freq']).delete(-1)
        self.predictions = self.forecaster.predict(steps=input_features['Windows to be forecasted'] * input_features['Window size'], last_window=self.data[input_features['Forecasted_param']].loc[input_features['Last-observed-window']]).to_frame().set_index(Newindex)

    def Generate_interval_prediction(self,input_features):
        """
        Generate_interval_prediction(self,input_features)
        
        This function outputs three sets of values (a lower bound, an upper bound and the most likely value), using a recursive multi-step probabilistic forecasting method.
        The confidence level can be set in the function parameters as "interval = [10, 90]".
        
        input_features is a dictionary. To find an example of its format refer to the ReadData.py file
        """

        # Create a time-index for the dates that are being predicted
        Newindex = pd.date_range(start=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=1), end=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=input_features['Windows to be forecasted']+1),freq=input_features['data-freq']).delete(-1)
        
        # [10 90] considers 80% (90-10) confidence interval ------- n_boot: Number of bootstrapping iterations used to estimate prediction intervals.
        self.interval_predictions = self.forecaster.predict_interval(steps=input_features['Windows to be forecasted'] * input_features['Window size'], interval = [10, 90],n_boot = 1000, last_window=self.data[input_features['Forecasted_param']].loc[input_features['Last-observed-window']]).set_index(Newindex)

    @staticmethod
    def Generate_disaggregation_regression():

        """
        Generate_disaggregation_regression()
        
        This function uses a linear regression model to disaggregate the electricity demand from PV generation in the data. 
        Note that the active power stored in the data is recorded at at each \textit{nmi} connection point to the grid and thus 
        is summation all electricity usage and generation that happens behind the meter. 
        """

        pv = [ Ridge(alpha=1.0).fit( np.array([customers[i].data.pv_system_size[datetimes[0]]/customers[i].data.active_power.max()  for i in customers_nmi_with_pv]).reshape((-1,1)),
                               np.array([customers[i].data.active_power[datetimes[t]]/customers[i].data.active_power.max() for i in customers_nmi_with_pv])     
                              ).coef_[0]   for t in range(0,len(datetimes))]

        for i in customers_nmi_with_pv:
            customers[i].data['pv_disagg'] = [ pv[t]*customers[i].data.pv_system_size[datetimes[0]] + min(customers[i].data.active_power[datetimes[t]] + pv[t]*customers[i].data.pv_system_size[datetimes[0]],0) for t in range(0,len(datetimes))]
            customers[i].data['demand_disagg'] = [max(customers[i].data.active_power[datetimes[t]] + pv[t]*customers[i].data.pv_system_size[datetimes[0]],0) for t in range(0,len(datetimes))]

        for i in list(set(customers_nmi) - set(customers_nmi_with_pv)):
            customers[i].data['pv_disagg'] = 0
            customers[i].data['demand_disagg'] = customers[customers_nmi[0]].data.active_power.values

    @staticmethod
    def Generate_disaggregation_optimisation():

        """
        Generate_disaggregation_optimisation()
        
        This function disaggregates the demand and generation for all the nodes in the system and all the time-steps, and adds the disaggergations to each
        class variable. It applies the disaggregation to all nmis. This fuction uses function "pool_executor_disaggregation" to run the disaggregation algorithm.  
        """
        Times = range(0,len(datetimes))
        # Times = range(0,len(data.loc[customers_nmi[0]][input_features['Start training']:input_features['Last-observed-window']]))
        result_disaggregation = pool_executor_disaggregation(Demand_disaggregation,Times)
        Total_res = [res for res in result_disaggregation]
        
        for i in customers_nmi_with_pv:
            customers[i].data['pv_disagg'] = [Total_res[t][0][i] for t in Times]
            customers[i].data['demand_disagg'] = [Total_res[t][1][i] for t in Times]

        for i in list(set(customers_nmi) - set(customers_nmi_with_pv)):
            customers[i].data['pv_disagg'] = 0
            customers[i].data['demand_disagg'] = customers[customers_nmi[0]].data.active_power.values

    #######
    # To be added
    #######
    # # This function outputs the forecasts using a Recursive multi-step point-forecasting method of each nmi individually considering reactive power an exogenous variable 
    # def Generate_forecaster_object_with_exogenous(self,input_features):
    #     # Create a forecasting object
    #     self.forecaster = ForecasterAutoreg(
    #             regressor = make_pipeline(StandardScaler(), Ridge()),
    #             lags      = input_features['Window size']      # The used data set has a 30-minute resolution. So, 48 denotes one full day window
    #         )

    #     # Train the forecaster using the train data
    #     self.forecaster.fit(y=self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']],
    #                         exog = self.data.loc[input_features['Start training']:input_features['End training']].reactive_power) 

    # def Generate_predictio_with_exogenous(self,input_features):
    #     # Generate predictions using normal forecasting
    #     Newindex = pd.date_range(start=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=1), end=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=input_features['Windows to be forecasted']+1),freq='30T').delete(-1)
    #     self.predictions = self.forecaster.predict(steps = input_features['Windows to be forecasted'] * input_features['Window size'], 
    #                                                last_window = self.data[input_features['Forecasted_param']].loc[input_features['Last-observed-window']],
    #                                                exog = self.data.reactive_power.loc[input_features['Last-observed-window']]
    #                                                 ).to_frame().set_index(Newindex)



# Create an instance for each nmi in the customers_class class
customers = {}
for customer in customers_nmi:
    customers[customer] = customers_class(customer,input_features)


def Demand_disaggregation(t):
    
    """
    Demand_disaggregation(t), where t is the time-step of the disaggregation.
    
    This function disaggregates the demand and generation for all the nodes in the system at time-step t. 

    It is uses an optimisation algorithm with constrain:
        P_{t}^{pv} * PanleSize_{i} + P^{d}_{i,t}  == P^{agg}_{i,t} + P^{pen-p}_{i,t} - P^{pen-n}_{i,t},
    with the objective:
        min (P_{t}^{pv} + 10000 * \sum_{i} (P^{pen-p}_{i,t} - P^{pen-n}_{i,t}) 
    variables P^{pen-p}_{i,t} and P^{pen-n}_{i,t}) are defined to prevenet infeasibilities the optimisation problem, and are added to the objective function
    with a big coefficient. Variables P_{t}^{pv} and P^{d}_{i,t} denote the irradiance at time t, and demand at nmi i and time t, respectively. Also, parameters 
    PanleSize_{i} and P^{agg}_{i,t} denote the PV panel size of nmi i, and the recorded aggregated demand at nmi i and time t, respectively.
    """

    Time = range(t,t+1)
    model=ConcreteModel()
    model.pv=Var(Time, bounds=(0,1))
    model.demand=Var(Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_p=Var(Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_n=Var(Time,customers_nmi_with_pv,within=NonNegativeReals)

    # # Constraints
    model.Const=ConstraintList()

    for t in Time:
        for i in customers_nmi_with_pv:
            model.Const.add(model.demand[t,i] - model.pv[t] * customers[i].data.pv_system_size[datetimes[0]] == customers[i].data.active_power[datetimes[t]] + model.penalty_p[t,i] - model.penalty_n[t,i]   )

    # # Objective
    def Objrule(model):
        return sum(model.pv[t] for t in Time) + 10000 * sum( sum( model.penalty_p[t,i] + model.penalty_n[t,i] for i in customers_nmi_with_pv ) for t in Time)
    model.obj=Objective(rule=Objrule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)
    

    print(" Disaggregating {first}-th time step".format(first = t))
    # print(t)

    return ({i:    - (model.pv[t].value * customers[i].data.pv_system_size[0] + model.penalty_p[t,i].value)  for i in customers_nmi_with_pv},
            {i:      model.demand[t,i].value + model.penalty_n[t,i].value  for i in customers_nmi_with_pv} )


# This function is used to parallelised the forecasting for each nmi
def pool_executor_disaggregation(function_name,Times):
    with ProcessPoolExecutor(max_workers=int(cpu_count()/core_usage),mp_context=mp.get_context('fork')) as executor:
        results = executor.map(function_name,Times)  
    return results


# # ==================================================================================================
# # Method (1): Recursive multi-step point-forecasting method
# # ==================================================================================================

# This function is used to parallelised the forecasting for each nmi
def pool_executor_forecast_pointbased(function_name,customers_nmi,input_features):
    
    """
    pool_executor_forecast_pointbased(function_name,customers_nmi,input_features)

    This functions (along with function run_prallel_forecast_pointbased) are used to parallelised forecast_pointbased() function for each nmi. It accepts the function name to be ran, the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the ReadData.py file.
    """

    with ProcessPoolExecutor(max_workers=int(cpu_count()/core_usage),mp_context=mp.get_context('fork')) as executor:
        results = executor.map(function_name,customers_nmi,repeat(input_features))  
    return results

# This function outputs the forecasting for each nmi
def run_prallel_forecast_pointbased(customers_nmi,input_features):

    """
    run_prallel_forecast_pointbased(customers_nmi,input_features)

    This functions (along with function pool_executor_forecast_pointbased) are used to parallelised forecast_pointbased() function for each nmi. It accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the ReadData.py file.
    """

    # from Load_Forecasting import customers 
    print(" Customer nmi: {first} --------> This is the {second}-th out of {third} customers".format(first = customers_nmi, second=list(customers.keys()).index(customers_nmi),third = len(customers)))

    # Train a forecasting object
    customers[customers_nmi].Generate_forecaster_object(input_features)
    
    # Generate predictions 
    customers[customers_nmi].Generate_prediction(input_features)
    
    return customers[customers_nmi].predictions.rename(columns={'pred': customers_nmi}) 

# This function uses the parallelised function and save the result into a single dictionary 
def forecast_pointbased(customers_nmi,input_features):
    
    """
    forecast_pointbased(customers_nmi,input_features) 

    This function generates prediction values for all the nmis using a recursive multi-step point-forecasting method. It uses function pool_executor_forecast_pointbased to generate
    the predictions for each nmi parallely (each on a separate core). This function accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the ReadData.py file.

    This function return the forecasted value of the desired parameter specified in the input_feature['Forecasted_param'] for the dates specified in the input_feature dictionary for 
    all the nmis in pandas.Dataframe format.
    """
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
    """
    pool_executor_forecast_interval(function_name,customers_nmi,input_features)

    This functions (along with function run_prallel_Interval_Load_Forecast) are used to parallelised forecast_interval() function for each nmi. It accepts the function name to be ran, the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the ReadData.py file.
    """
    with ProcessPoolExecutor(max_workers=int(cpu_count()/core_usage),mp_context=mp.get_context('fork')) as executor:
        results = executor.map(function_name,customers_nmi,repeat(input_features))  
    return results

# This function outputs the forecasting for each nmi
def run_prallel_Interval_Load_Forecast(customers_nmi,input_features):

    """
    run_prallel_Interval_Load_Forecast(customers_nmi,input_features)

    This functions (along with function pool_executor_forecast_interval) are used to parallelised forecast_interval() function for each nmi. It accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the ReadData.py file.
    """

    # from Load_Forecasting import customers 
    print(" Customer nmi: {first} --------> This is the {second}-th out of {third} customers".format(first = customers_nmi, second=list(customers.keys()).index(customers_nmi),third = len(customers)))



    # Train a forecasting object
    customers[customers_nmi].Generate_forecaster_object(input_features)
    
    # Generate interval predictions 
    customers[customers_nmi].Generate_interval_prediction(input_features)
    
    return customers[customers_nmi].interval_predictions.rename(columns={'pred': customers_nmi})


# This function uses the parallelised function and save the result into a single dictionary 
def forecast_interval(customers_nmi,input_features):

    """
    forecast_pointbased(customers_nmi,input_features) 

    This function generates prediction values for all the nmis using a recursive multi-step probabilistic forecasting method. It uses function pool_executor_forecast_interval to generate
    the predictions for each nmi parallely (each on a separate core). This function accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the ReadData.py file.

    This function return the forecasted values for the lower bound, upper bound and the most likely values of the desired parameter specified in the input_feature['Forecasted_param'] for the dates specified in the input_feature dictionary for 
    all the nmis in pandas.Dataframe format.
    """
    
    predictions_prallel = pool_executor_forecast_interval(run_prallel_Interval_Load_Forecast,customers_nmi,input_features)

    # Aggregate the results from the parallelised function into a dictionary
    predictions_output_interval = {}
    for res in predictions_prallel:
        predictions_output_interval[int(res.columns[0])] = res.rename(columns={res.columns[0]: 'pred'})
    
    return predictions_output_interval


# Export interval based method into a json file
def export_interval_result_to_json(predictions_output_interval):
    """
    export_interval_result_to_json(predictions_output_interval)

    This function saves the predictions generated by function forecast_interval as a json file.
    """


    copy_predictions_output = copy(predictions_output_interval)
    for c in copy_predictions_output.keys():
        copy_predictions_output[c] = json.loads(copy_predictions_output[c].to_json())
    with open("prediction_interval_based.json","w") as f:
        json.dump(copy_predictions_output,f)

def read_json_interval(filename):

    """
    read_json_interval(filename)

    This function imports the json file generated by function export_interval_result_to_json
    and return the saved value in pandas.Dataframe format.
    """
    with open(filename,"r") as f:
        loaded_predictions_output = json.load(f)

    for l in list(loaded_predictions_output.keys()):
        loaded_predictions_output[int(l)] = pd.read_json(json.dumps(loaded_predictions_output[l]))
        del loaded_predictions_output[l]
    
    return(loaded_predictions_output)

def Forecast_using_disaggregation(customers_nmi,input_features):

    """
    Forecast_using_disaggregation(customers_nmi,input_features)

    This function is used to generate forecast values. It first disaggregates the demand and generation for all nmi using function
    Generate_disaggregation_optimisation. It then uses function forecast_pointbased for the disaggregated demand and generation and produces separate forecast. It finally sums up the two values
    and returns an aggregated forecast for all nmis in pandas.Dataframe format.
    """
    
    customers_class.Generate_disaggregation_optimisation()

    input_features_copy = copy(input_features)
    input_features_copy['Forecasted_param']= 'pv_disagg'
    predictions_output_pv = forecast_pointbased(customers_nmi,input_features_copy)

    input_features_copy['Forecasted_param']= 'demand_disagg'
    predictions_output_demand = forecast_pointbased(customers_nmi,input_features_copy)

    predictions_agg = predictions_output_demand + predictions_output_pv

    return(predictions_agg)





# import matplotlib.pyplot as plt
# customers_class.Generate_disaggregation_optimisation()
# customers2 = copy(customers)
# customers_class.Generate_disaggregation_regression()

# nmi = customers_nmi_with_pv[5]

# fig, ax = plt.subplots(figsize=(12, 3.5))
# customers[nmi].data.active_power[48*5:48*6].plot(linewidth=2, label='main', ax=ax)
# customers2[nmi].data.pv_disagg[48*5:48*6].plot(linewidth=2, label='pv optimisation', ax=ax)
# customers[nmi].data.pv_disagg[48*5:48*6].plot(linewidth=2, label='pv regression', ax=ax)
# ax.legend()
# plt.show()   
