from operator import le
import pandas as pd
import numpy as np
import pickle
import copy
from sklearn.linear_model import Ridge # Linear least squares with l2 regularization. (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
from sklearn.pipeline import make_pipeline # Construct a Pipeline from the given estimators (for more information: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance.
from skforecast.ForecasterAutoreg import ForecasterAutoreg # A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
from skforecast.model_selection import grid_search_forecaster, backtesting_forecaster
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

from itertools import repeat
from multiprocess import cpu_count
import multiprocess as mp
from pyomo.environ import NonNegativeReals, ConstraintList, ConcreteModel, Var, Objective, Set, Constraint
from pyomo.opt import SolverFactory
from tqdm import tqdm
from functools import partialmethod
import json
from copy import deepcopy as copy

# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# from ReadData_init import input_features,customers_nmi,customers_nmi_with_pv,datetimes, customers



def read_data(input_features):
    if input_features['file_type'] == 'Converge':

        # Read data
        data = pd.read_csv(input_features['file_name'])

        # # ###### Pre-process the data ######

        # format datetime to pandas datetime format
        data['datetime'] = pd.to_datetime(data['datetime'])

        # Add weekday column to the data
        data['DayofWeek'] = data['datetime'].dt.day_name()

        # Save customer nmis in a list
        customers_nmi = list(dict.fromkeys(data['nmi'].values.tolist()))

        # *** Temporary *** the last day of the data (2022-07-31)
        # is very different from the rest, and is ommitted for now.
        filt = (data['datetime'] < '2022-07-31')
        data = data.loc[filt].copy()

        # Make datetime index of the dataset
        data.set_index(['nmi', 'datetime'], inplace=True)

        # save unique dates of the data
        datetimes = data.index.unique('datetime')

        # To obtain the data for each nmi: --> data.loc[nmi]
        # To obtain the data for timestep t: --> data.loc[pd.IndexSlice[:, datetimes[t]], :]


        # Add PV instalation and size, and load type to the data from nmi.csv file
        # ==============================================================================
        # nmi_available = [i for i in customers_nmi if (data_nmi['nmi'] ==  i).any()] # use this line if there are some nmi's in the network that are not available in the nmi.csv file
        data_nmi = pd.read_csv('nmi.csv')
        data_nmi.set_index(data_nmi['nmi'],inplace=True)

        import itertools
        customers_nmi_with_pv = [ data_nmi.loc[i]['nmi'] for i in customers_nmi if data_nmi.loc[i]['has_pv']==True ]
        data['has_pv']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['has_pv']] for i in customers_nmi]* len(datetimes)))
        data['customer_kind']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['customer_kind']] for i in customers_nmi]* len(datetimes)))
        data['pv_system_size']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['pv_system_size']] for i in customers_nmi]* len(datetimes)))

        # # This line is added to prevent the aggregated demand from being negative when there is not PV
        # # Also, from the data, it seems that negative sign is a mistake and the positive values make more sense in those nmis
        # # for i in customers_nmi:
        # #     if data.loc[i].pv_system_size[0] == 0:
        # #         data.at[i,'active_power'] = data.loc[i].active_power.abs()

    elif input_features['file_type'] == 'NextGen':
        with open(input_features['file_name'], 'rb') as handle:
            data = pickle.load(handle)
        data = data[~data.index.duplicated(keep='first')]

        datetimes = data.loc[data.index[0][0]].index
        customers_nmi = list(data.loc[pd.IndexSlice[:, datetimes[0]], :].index.get_level_values('nmi'))
        customers_nmi_with_pv = copy(customers_nmi)

        # To obtain the data for each nmi: --> data.loc[nmi]
        # To obtain the data for timestep t: --> data.loc[pd.IndexSlice[:, datetimes[t]], :]

        ##### Read 5-minute weather data from SolCast
        data_weather = pd.read_csv('-35.3075_149.124417_Solcast_PT5M.csv')

        data_weather['PeriodStart'] = pd.to_datetime(data_weather['PeriodStart'])
        data_weather = data_weather.drop('PeriodEnd', axis=1)
        data_weather = data_weather.rename(columns={"PeriodStart": "datetime"})
        data_weather.set_index('datetime', inplace=True)
        data_weather.index = data_weather.index.tz_convert('Australia/Sydney')

        # *** Temporary *** the last day of the data (2022-07-31)
        # is very different from the rest, and is ommitted for now.
        filt = (data_weather.index > '2018-01-01 23:59:00')
        data_weather = data_weather.loc[filt].copy()

    global customers_class

    class customers_class:
        
        num_of_customers = 0

        def __init__(self, nmi,input_features):

            self.nmi = nmi      # store nmi in each object              
            self.data = data.loc[self.nmi]      # store data in each object         

            customers_class.num_of_customers += 1

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
                            steps       =  input_features['Window size'],
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
            
            Newindex = pd.date_range(start=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=1), end=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=input_features['Windows to be forecasted']+1),freq=input_features['data_freq']).delete(-1)
            self.predictions = self.forecaster.predict(steps=input_features['Windows to be forecasted'] * input_features['Window size'], last_window=self.data[input_features['Forecasted_param']].loc[input_features['Last-observed-window']]).to_frame().set_index(Newindex)

        def Generate_interval_prediction(self,input_features):
            """
            Generate_interval_prediction(self,input_features)
            
            This function outputs three sets of values (a lower bound, an upper bound and the most likely value), using a recursive multi-step probabilistic forecasting method.
            The confidence level can be set in the function parameters as "interval = [10, 90]".
            
            input_features is a dictionary. To find an example of its format refer to the ReadData.py file
            """

            # Create a time-index for the dates that are being predicted
            Newindex = pd.date_range(start=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=1), end=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=input_features['Windows to be forecasted']+1),freq=input_features['data_freq']).delete(-1)
            
            # [10 90] considers 80% (90-10) confidence interval ------- n_boot: Number of bootstrapping iterations used to estimate prediction intervals.
            self.interval_predictions = self.forecaster.predict_interval(steps=input_features['Windows to be forecasted'] * input_features['Window size'], interval = [10, 90],n_boot = 1000, last_window=self.data[input_features['Forecasted_param']].loc[input_features['Last-observed-window']]).set_index(Newindex)


        def Generate_disaggregation_using_reactive(self):

            QP_coeff = (self.data.load_reactive/self.data.load_active[self.data.load_active > 0.001]).resample('D').mean()
            QP_coeff[(QP_coeff.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")] = QP_coeff[-1]
            QP_coeff = QP_coeff.resample(input_features['data_freq']).ffill()
            QP_coeff = QP_coeff.drop(QP_coeff.index[-1])
            QP_coeff = QP_coeff[QP_coeff.index <= self.data.load_reactive.index[-1]]

            set_diff = list( set(QP_coeff.index)-set(self.data.load_reactive.index) )
            QP_coeff = QP_coeff.drop(set_diff)

            load_est = self.data.load_reactive / QP_coeff 
            pv_est = load_est  - self.data.active_power
            pv_est[pv_est < 0] = 0
            load_est = pv_est + self.data.active_power
            
            self.data['pv_disagg'] = pv_est
            self.data['demand_disagg'] = load_est

    customers = {customer: customers_class(customer,input_features) for customer in customers_nmi}

    return data, customers_nmi,customers_nmi_with_pv,datetimes, customers

# # This function is used to parallelised the forecasting for each nmi
# def pool_executor_parallel(function_name,repeat_iter,input_features):
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         results = list(executor.map(function_name,repeat_iter,repeat(input_features)))  
#     return results

def pool_executor_parallel(function_name,repeat_iter,input_features):
        with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
            results = list(executor.map(function_name,repeat_iter,repeat(input_features)))  
        return results


# # ==================================================================================================
# # Method (1): Recursive multi-step point-forecasting method
# # ==================================================================================================

# This function outputs the forecasting for each nmi
def run_single_forecast_pointbased(customers,input_features):

    """
    run_prallel_forecast_pointbased(customers_nmi,input_features)

    This functions (along with function pool_executor_forecast_pointbased) are used to parallelised forecast_pointbased() function for each nmi. It accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the ReadData.py file.
    """
    
    # print(customers)
    print(" Customer nmi: {first}".format(first = customers.nmi))

    # Train a forecasting object
    customers.Generate_forecaster_object(input_features)
    
    # Generate predictions 
    customers.Generate_prediction(input_features)

    return customers.predictions.rename(columns={'pred': customers.nmi})


def forecast_pointbased(customers,input_features):

    predictions_prallel = pool_executor_parallel(run_single_forecast_pointbased,customers.values(),input_features)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=1)

    return predictions_prallel

# # ==================================================================================================
# # Method (2): Recursive multi-step probabilistic forecasting method
# # ==================================================================================================

# This function outputs the forecasting for each nmi
def run_single_Interval_Load_Forecast(customers,input_features):

    """
    run_prallel_Interval_Load_Forecast(customers_nmi,input_features)

    This functions (along with function pool_executor_forecast_interval) are used to parallelised forecast_interval() function for each nmi. It accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the ReadData.py file.
    """

    print(" Customer nmi: {first}".format(first = customers.nmi))


    # Train a forecasting object
    customers.Generate_forecaster_object(input_features)
    
    # Generate interval predictions 
    customers.Generate_interval_prediction(input_features)
    
    return customers.interval_predictions.rename(columns={'pred': customers.nmi})


# This function uses the parallelised function and save the result into a single dictionary 
def forecast_interval(customers,input_features):

    """
    forecast_interval(customers_nmi,input_features) 

    This function generates prediction values for all the nmis using a recursive multi-step probabilistic forecasting method. It uses function pool_executor_forecast_interval to generate
    the predictions for each nmi parallely (each on a separate core). This function accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the ReadData.py file.

    This function return the forecasted values for the lower bound, upper bound and the most likely values of the desired parameter specified in the input_feature['Forecasted_param'] for the dates specified in the input_feature dictionary for 
    all the nmis in pandas.Dataframe format.
    """

    predictions_prallel = pool_executor_parallel(run_single_Interval_Load_Forecast,customers.values(),input_features)
 
    predictions_prallel = {predictions_prallel[i].columns[0]: predictions_prallel[i].rename(columns={predictions_prallel[i].columns[0]: 'pred'}) for i in range(0,len(predictions_prallel))}

    return predictions_prallel



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
        loaded_predictions_output[l] = pd.read_json(json.dumps(loaded_predictions_output[l]))
    
    return(loaded_predictions_output)

# # ==================================================================================================
# # Disaggregations algorithms
# # ==================================================================================================


def run_single_demand_disaggregation_optimisation(time_step,customers_nmi_with_pv,datetimes,data_one_time):

    """
    Demand_disaggregation(t,customers_nmi_with_pv,customers), where t is the time-step of the disaggregation.
    
    This function disaggregates the demand and generation for all the nodes in the system at time-step t. 

    It is uses an optimisation algorithm with constrain:
        P_{t}^{pv} * PanleSize_{i} + P^{d}_{i,t}  == P^{agg}_{i,t} + P^{pen-p}_{i,t} - P^{pen-n}_{i,t},
    with the objective:
        min (P_{t}^{pv} + 10000 * \sum_{i} (P^{pen-p}_{i,t} - P^{pen-n}_{i,t}) 
    variables P^{pen-p}_{i,t} and P^{pen-n}_{i,t}) are defined to prevenet infeasibilities the optimisation problem, and are added to the objective function
    with a big coefficient. Variables P_{t}^{pv} and P^{d}_{i,t} denote the irradiance at time t, and demand at nmi i and time t, respectively. Also, parameters 
    PanleSize_{i} and P^{agg}_{i,t} denote the PV panel size of nmi i, and the recorded aggregated demand at nmi i and time t, respectively.
    """

    t = time_step

    model=ConcreteModel()
    model.Time = Set(initialize=range(t,t+1))
    model.pv=Var(model.Time, bounds=(0,1))
    model.demand=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_p=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_n=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)

    # # Constraints
    def LoadBalance(model,t,i):
        return model.demand[t,i] - model.pv[t] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]] + model.penalty_p[t,i] - model.penalty_n[t,i] 
    model.cons = Constraint(model.Time,customers_nmi_with_pv,rule=LoadBalance)

    # # Objective
    def Objrule(model):
        return sum(model.pv[t] for t in model.Time) + 10000 * sum( sum( model.penalty_p[t,i] + model.penalty_n[t,i] for i in customers_nmi_with_pv ) for t in model.Time)
    model.obj=Objective(rule=Objrule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    print(" Disaggregating {first}-th time step".format(first = t))
    # print(t)

    result_output_temp =  ({i:    - (model.pv[t].value * data_one_time.loc[i].pv_system_size[0] + model.penalty_p[t,i].value)  for i in customers_nmi_with_pv},
            {i:      model.demand[t,i].value + model.penalty_n[t,i].value  for i in customers_nmi_with_pv} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)
    
    # result_output = pd.concat({datetimes[t]: result_output}, names=['datetime'])

    return result_output

def run_single_demand_disaggregation_optimisation_for_parallel(time_step,customers_nmi_with_pv,datetimes):

    """
    Demand_disaggregation(t,customers_nmi_with_pv,customers), where t is the time-step of the disaggregation.
    
    This function disaggregates the demand and generation for all the nodes in the system at time-step t. 

    It is uses an optimisation algorithm with constrain:
        P_{t}^{pv} * PanleSize_{i} + P^{d}_{i,t}  == P^{agg}_{i,t} + P^{pen-p}_{i,t} - P^{pen-n}_{i,t},
    with the objective:
        min (P_{t}^{pv} + 10000 * \sum_{i} (P^{pen-p}_{i,t} - P^{pen-n}_{i,t}) 
    variables P^{pen-p}_{i,t} and P^{pen-n}_{i,t}) are defined to prevenet infeasibilities the optimisation problem, and are added to the objective function
    with a big coefficient. Variables P_{t}^{pv} and P^{d}_{i,t} denote the irradiance at time t, and demand at nmi i and time t, respectively. Also, parameters 
    PanleSize_{i} and P^{agg}_{i,t} denote the PV panel size of nmi i, and the recorded aggregated demand at nmi i and time t, respectively.
    """

    t = time_step
    data_one_time = shared_data_disaggregation_optimisation.loc[pd.IndexSlice[:, datetimes[t]], :]

    model=ConcreteModel()
    model.Time = Set(initialize=range(t,t+1))
    model.pv=Var(model.Time, bounds=(0,1))
    model.demand=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_p=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_n=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)

    # # Constraints
    def LoadBalance(model,t,i):
        return model.demand[t,i] - model.pv[t] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]] + model.penalty_p[t,i] - model.penalty_n[t,i] 
    model.cons = Constraint(model.Time,customers_nmi_with_pv,rule=LoadBalance)

    # # Objective
    def Objrule(model):
        return sum(model.pv[t] for t in model.Time) + 10000 * sum( sum( model.penalty_p[t,i] + model.penalty_n[t,i] for i in customers_nmi_with_pv ) for t in model.Time)
    model.obj=Objective(rule=Objrule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)
    

    print(" Disaggregating {first}-th time step".format(first = t))
    # print(t)

    result_output_temp =  ({i:    - (model.pv[t].value * data_one_time.loc[i].pv_system_size[0] + model.penalty_p[t,i].value)  for i in customers_nmi_with_pv},
            {i:      model.demand[t,i].value + model.penalty_n[t,i].value  for i in customers_nmi_with_pv} )

    result_output = b = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)

    return result_output

def disaggregation_optimisation(data,input_features,datetimes,customers_nmi_with_pv):

    """
    Generate_disaggregation_optimisation()
    
    This function disaggregates the demand and generation for all the nodes in the system and all the time-steps, and adds the disaggergations to each
    class variable. It applies the disaggregation to all nmis. This fuction uses function "pool_executor_disaggregation" to run the disaggregation algorithm.  
    """

    def pool_executor_parallel_time(function_name,repeat_iter,customers_nmi_with_pv,datetimes,data,input_features):
        
        global shared_data_disaggregation_optimisation

        shared_data_disaggregation_optimisation = copy(data)
        with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
            results = list(executor.map(function_name,repeat_iter,repeat(customers_nmi_with_pv),repeat(datetimes)))  
        return results

    predictions_prallel = pool_executor_parallel_time(run_single_demand_disaggregation_optimisation_for_parallel,range(0,len(datetimes)),customers_nmi_with_pv,datetimes,data,input_features)
    
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    print('Done')

    # print(len(predictions_prallel))
    
    # if 'shared_data_disaggregation_optimisation' in globals():
    #     del(shared_data_disaggregation_optimisation)

    return predictions_prallel


def Generate_disaggregation_regression(customers,customers_nmi,customers_nmi_with_pv,datetimes):

    """
    Generate_disaggregation_regression(customers)
    
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


def Forecast_using_disaggregation(data,input_features,datetimes,customers_nmi_with_pv,customers_nmi,customers):

    """
    Forecast_using_disaggregation(customers_nmi,input_features)

    This function is used to generate forecast values. It first disaggregates the demand and generation for all nmi using function
    Generate_disaggregation_optimisation. It then uses function forecast_pointbased for the disaggregated demand and generation and produces separate forecast. It finally sums up the two values
    and returns an aggregated forecast for all nmis in pandas.Dataframe format.
    """
    
    opt_disaggregate = disaggregation_optimisation(data,input_features,datetimes,customers_nmi_with_pv)

    for i in customers_nmi_with_pv:
        customers[i].data['pv_disagg'] = opt_disaggregate.loc[i].pv_disagg.values
        customers[i].data['demand_disagg'] = opt_disaggregate.loc[i].pv_disagg.values

    for i in list(set(customers_nmi) - set(customers_nmi_with_pv)):
        customers[i].data['pv_disagg'] = 0
        customers[i].data['demand_disagg'] = customers[i].data.active_power.values


    input_features_copy = copy(input_features)
    input_features_copy['Forecasted_param']= 'pv_disagg'
    predictions_output_pv = forecast_pointbased(customers,input_features_copy)

    input_features_copy['Forecasted_param']= 'demand_disagg'
    predictions_output_demand = forecast_pointbased(customers,input_features_copy)

    predictions_agg = predictions_output_demand + predictions_output_pv

    return(predictions_agg)



def run_single_disaggregate_using_reactive(customers,input_features):

    customers.Generate_disaggregation_using_reactive()

    result = pd.DataFrame(customers.data.pv_disagg)
    result['demand_disagg'] = customers.data.demand_disagg
    nmi = [customers.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return(result)

def Generate_disaggregation_using_reactive_all(customers,input_features):

    predictions_prallel = pool_executor_parallel(run_single_disaggregate_using_reactive,customers.values(),input_features)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return(predictions_prallel)





# # Set features of the predections
# input_features = {  'file_type': 'Converge',
#                     'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
#                     'Start training': '2022-07-01',
#                     'End training': '2022-07-27',
#                     'Last-observed-window': '2022-07-27',
#                     'Window size': 48 ,
#                     'Windows to be forecasted':    3,     
#                     'data_freq' : '30T',
#                     'core_usage': 4       # 1/core_usage shows core percentage usage we want to use
#                      }

# # Set features of the predections
# input_features = {  'file_type': 'NextGen',
#                     'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
#                     'Start training': '2018-01-01',
#                     'End training': '2018-02-01',
#                     'Last-observed-window': '2018-02-01',
#                     'Window size':  288,
#                     'Windows to be forecasted':    3,
#                     'data_freq' : '5T',
#                     'core_usage': 6      }  

# # from ReadData_init import input_features
# [data, customers_nmi,customers_nmi_with_pv,datetimes, customers] = read_data(input_features)

