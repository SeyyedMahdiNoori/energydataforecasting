from operator import le
from sqlite3 import Time
import pandas as pd
import numpy as np
import pickle
import copy
from sklearn.linear_model import Ridge # Linear least squares with l2 regularization. (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
from sklearn.pipeline import make_pipeline # Construct a Pipeline from the given estimators (for more information: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance.
from sklearn.ensemble import RandomForestRegressor

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

# from read_data_init import input_features,customers_nmi,customers_nmi_with_pv,datetimes, customers



def read_data(input_features):
    if input_features['file_type'] == 'Converge':

        # Read data
        data = pd.read_csv(input_features['data_path'])

        # # ###### Pre-process the data ######

        # format datetime to pandas datetime format
        data['datetime'] = pd.to_datetime(data['datetime'])

        # Add weekday column to the data
        data['DayofWeek'] = data['datetime'].dt.day_name()

        # Save customer nmis in a list
        customers_nmi = list(dict.fromkeys(data['nmi'].values.tolist()))

        # # *** Temporary *** the last day of the data (2022-07-31)
        # # is very different from the rest, and is ommitted for now.
        # filt = (data['datetime'] < '2022-07-31')
        # data = data.loc[filt].copy()

        # Make datetime index of the dataset
        data.set_index(['nmi', 'datetime'], inplace=True)

        # save unique dates of the data
        datetimes = data.index.unique('datetime')

        # To obtain the data for each nmi: --> data.loc[nmi]
        # To obtain the data for timestep t: --> data.loc[pd.IndexSlice[:, datetimes[t]], :]


        # Add PV instalation and size, and load type to the data from nmi.csv file
        # ==============================================================================
        # nmi_available = [i for i in customers_nmi if (data_nmi['nmi'] ==  i).any()] # use this line if there are some nmi's in the network that are not available in the nmi.csv file
        data_nmi = pd.read_csv(input_features['nmi_type_path'])
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

        # # # TBA
        data_weather = {}
        
    elif input_features['file_type'] == 'NextGen':

        # with open(input_features['file_name'], 'rb') as handle:
        #     data = pickle.load(handle)
        # data.rename(columns={'load_reactive': 'reactive_power'},inplace=True)


        data = pd.read_csv(input_features['file_path'])
        # data = data[~data.index.duplicated(keep='first')]
        data.rename(columns={'load_reactive': 'reactive_power'},inplace=True)
        
        # format datetime to pandas datetime format
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Make datetime index of the dataset
        data.set_index(['nmi', 'datetime'], inplace=True)

        datetimes = data.loc[data.index[0][0]].index
        customers_nmi = list(data.loc[pd.IndexSlice[:, datetimes[0]], :].index.get_level_values('nmi'))
        customers_nmi_with_pv = copy(customers_nmi)

        # To obtain the data for each nmi: --> data.loc[nmi]
        # To obtain the data for timestep t: --> data.loc[pd.IndexSlice[:, datetimes[t]], :]

        ##### Read 5-minute weather data from SolCast for three locations
        data_weather0 = pd.read_csv(input_features['weather_data1_path'])
        data_weather0['PeriodStart'] = pd.to_datetime(data_weather0['PeriodStart'])
        data_weather0 = data_weather0.drop('PeriodEnd', axis=1)
        data_weather0 = data_weather0.rename(columns={"PeriodStart": "datetime"})
        data_weather0.set_index('datetime', inplace=True)
        data_weather0.index = data_weather0.index.tz_convert('Australia/Sydney')
        # data_weather0['isweekend'] = (data_weather0.index.day_of_week > 4).astype(int)
        # data_weather0['Temp_EWMA'] = data_weather0.AirTemp.ewm(com=0.5).mean()
        
        # *** Temporary *** 
        filt = (data_weather0.index > '2018-01-01 23:59:00')
        data_weather0 = data_weather0.loc[filt].copy()
        
        data_weather1 = pd.read_csv(input_features['weather_data2_path'])
        data_weather1['PeriodStart'] = pd.to_datetime(data_weather1['PeriodStart'])
        data_weather1 = data_weather1.drop('PeriodEnd', axis=1)
        data_weather1 = data_weather1.rename(columns={"PeriodStart": "datetime"})
        data_weather1.set_index('datetime', inplace=True)
        data_weather1.index = data_weather1.index.tz_convert('Australia/Sydney')


        # *** Temporary *** 
        filt = (data_weather1.index > '2018-01-01 23:59:00')
        data_weather1 = data_weather1.loc[filt].copy()

        data_weather2 = pd.read_csv(input_features['weather_data3_path'])
        data_weather2['PeriodStart'] = pd.to_datetime(data_weather2['PeriodStart'])
        data_weather2 = data_weather2.drop('PeriodEnd', axis=1)
        data_weather2 = data_weather2.rename(columns={"PeriodStart": "datetime"})
        data_weather2.set_index('datetime', inplace=True)
        data_weather2.index = data_weather2.index.tz_convert('Australia/Sydney')


        # *** Temporary *** 
        filt = (data_weather2.index > '2018-01-01 23:59:00')
        data_weather2 = data_weather2.loc[filt].copy()

        data_weather = {'Loc1': data_weather0,
                        'Loc2': data_weather1,
                        'Loc3': data_weather2,  }


    global Customers

    class Customers:
        
        num_of_customers = 0

        def __init__(self, nmi,input_features):

            self.nmi = nmi      # store nmi in each object              
            self.data = data.loc[self.nmi]      # store data in each object         

            Customers.num_of_customers += 1

        def generate_forecaster(self,input_features):
            
            """
            generate_forecaster(self,input_features)
            
            This function generates a forecaster object to be used for a recursive multi-step forecasting method. 
            It is based on a linear least squares with l2 regularization method. Alternatively, LinearRegression() and Lasso() that
            have different objective can be used with the same parameters.
            
            input_features is a dictionary. To find an example of its format refer to the read_data.py file
            """

            # Create a forecasting object
            self.forecaster = ForecasterAutoreg(
                    regressor = make_pipeline(StandardScaler(), Ridge()),  
                    lags      = input_features['Window size']      # The used data set has a 30-minute resolution. So, 48 denotes one full day window
                )

            # Train the forecaster using the train data
            self.forecaster.fit(y=self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']])

        def generate_optimised_forecaster_object(self,input_features):
            
            """
            generate_optimised_forecaster_object(self,input_features)
            
            This function generates a forecaster object for each \textit{nmi} to be used for a recursive multi-step forecasting method.
            It builds on function Generate\_forecaster\_object by combining grid search strategy with backtesting to identify the combination of lags 
            and hyperparameters that achieve the best prediction performance. As default, it is based on a linear least squares with \textit{l2} regularisation method. 
            Alternatively, it can use LinearRegression() and Lasso() methods to generate the forecaster object.

            input_features is a dictionary. To find an example of its format refer to the read_data.py file
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
            

        def generate_prediction(self,input_features):
            """
            generate_prediction(self,input_features)
            
            This function outputs the prediction values using a Recursive multi-step point-forecasting method. 
            
            input_features is a dictionary. To find an example of its format refer to the read_data.py file
            """
            
            new_index = pd.date_range(start=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=1), end=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=input_features['Windows to be forecasted']+1),freq=input_features['data_freq']).delete(-1)
            self.predictions = self.forecaster.predict(steps=input_features['Windows to be forecasted'] * input_features['Window size'], last_window=self.data[input_features['Forecasted_param']].loc[input_features['Last-observed-window']]).to_frame().set_index(new_index)

        def generate_interval_prediction(self,input_features):
            """
            generate_interval_prediction(self,input_features)
            
            This function outputs three sets of values (a lower bound, an upper bound and the most likely value), using a recursive multi-step probabilistic forecasting method.
            The confidence level can be set in the function parameters as "interval = [10, 90]".
        
            input_features is a dictionary. To find an example of its format refer to the read_data.py file
            """

            # Create a time-index for the dates that are being predicted
            new_index = pd.date_range(start=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=1), end=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=input_features['Windows to be forecasted']+1),freq=input_features['data_freq']).delete(-1)
            
            # [10 90] considers 80% (90-10) confidence interval ------- n_boot: Number of bootstrapping iterations used to estimate prediction intervals.
            self.interval_predictions = self.forecaster.predict_interval(steps=input_features['Windows to be forecasted'] * input_features['Window size'], interval = [10, 90],n_boot = 1000, last_window=self.data[input_features['Forecasted_param']].loc[input_features['Last-observed-window']]).set_index(new_index)


        def generate_disaggregation_using_reactive(self):

            QP_coeff = (self.data.reactive_power.between_time('0:00','5:00')/self.data.active_power.between_time('0:00','5:00')[self.data.active_power.between_time('0:00','5:00') > 0.001]).resample('D').mean()
            QP_coeff[(QP_coeff.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")] = QP_coeff[-1]
            QP_coeff = QP_coeff.resample(input_features['data_freq']).ffill()
            QP_coeff = QP_coeff.drop(QP_coeff.index[-1])
            QP_coeff = QP_coeff[QP_coeff.index <= self.data.reactive_power.index[-1]]

            set_diff = list( set(QP_coeff.index)-set(self.data.reactive_power.index) )
            QP_coeff = QP_coeff.drop(set_diff)

            load_est = self.data.reactive_power / QP_coeff 
            pv_est = load_est  - self.data.active_power
            pv_est[pv_est < 0] = 0
            # pv_est = pv_est[~pv_est.index.duplicated(keep='first')]
            load_est = pv_est + self.data.active_power
            
            self.data['pv_disagg'] = pv_est
            self.data['demand_disagg'] = load_est

        def Generate_disaggregation_positive_minimum_PV(self):
            D = copy(self.data.active_power)
            D[D<=0] = 0
            S = copy(self.data.active_power)
            S[S>=0] = 0
            self.data['pv_disagg'] =  - S
            self.data['demand_disagg'] = D

    customers = {customer: Customers(customer,input_features) for customer in customers_nmi}

    return data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather

# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Solar and Demand Forecasting functions
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================

# # This function is used to parallelised the forecasting for each nmi
# def pool_executor_parallel(function_name,repeat_iter,input_features):
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         results = list(executor.map(function_name,repeat_iter,repeat(input_features)))  
#     return results

def pool_executor_parallel(function_name,repeat_iter,input_features):
        with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
            results = list(executor.map(function_name,repeat_iter,repeat(input_features)))  
        return results


# # ================================================================
# # Method (1): Recursive multi-step point-forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_pointbased_single_node(customer,input_features):

    """
    run_prallel_forecast_pointbased(customers_nmi,input_features)

    This functions (along with function pool_executor_forecast_pointbased) are used to parallelised forecast_pointbased() function for each nmi. It accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the read_data.py file.
    """
    
    # print(customers)
    print(" Customer nmi: {first}".format(first = customer.nmi))

    # Train a forecasting object
    customer.generate_forecaster(input_features)
    
    # Generate predictions 
    customer.generate_prediction(input_features)

    return customer.predictions.rename(columns={'pred': customer.nmi})


def forecast_pointbased_multiple_nodes(customers,input_features):

    predictions_prallel = pool_executor_parallel(forecast_pointbased_single_node,customers.values(),input_features)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=1)

    return predictions_prallel

# # ================================================================
# # Method (2): Recursive multi-step probabilistic forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_inetervalbased_single_node(customer,input_features):

    """
    run_prallel_Interval_Load_Forecast(customers_nmi,input_features)

    This functions (along with function pool_executor_forecast_interval) are used to parallelised forecast_interval() function for each nmi. It accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the read_data.py file.
    """

    print(" Customer nmi: {first}".format(first = customer.nmi))


    # Train a forecasting object
    customer.generate_forecaster(input_features)
    
    # Generate interval predictions 
    customer.generate_interval_prediction(input_features)
    
    return customer.interval_predictions


# This function outputs the forecasting for each nmi
def forecast_inetervalbased_single_node_for_parallel(customer,input_features):

    """
    run_prallel_Interval_Load_Forecast(customers_nmi,input_features)

    This functions (along with function pool_executor_forecast_interval) are used to parallelised forecast_interval() function for each nmi. It accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the read_data.py file.
    """

    print(" Customer nmi: {first}".format(first = customer.nmi))


    # Train a forecasting object
    customer.generate_forecaster(input_features)
    
    # Generate interval predictions 
    customer.generate_interval_prediction(input_features)
    
    return customer.interval_predictions.rename(columns={'pred': customer.nmi})



# This function uses the parallelised function and save the result into a single dictionary 
def forecast_inetervalbased_multiple_nodes(customers,input_features):

    """
    forecast_interval(customers_nmi,input_features) 

    This function generates prediction values for all the nmis using a recursive multi-step probabilistic forecasting method. It uses function pool_executor_forecast_interval to generate
    the predictions for each nmi parallely (each on a separate core). This function accepts the list "customers_nmi", and the dictionary 
    "input_features" as inputs. Examples of the list and the dictionary used in this function can be found in the read_data.py file.

    This function return the forecasted values for the lower bound, upper bound and the most likely values of the desired parameter specified in the input_feature['Forecasted_param'] for the dates specified in the input_feature dictionary for 
    all the nmis in pandas.Dataframe format.
    """

    predictions_prallel = pool_executor_parallel(forecast_inetervalbased_single_node_for_parallel,customers.values(),input_features)
 
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

# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Solar and Demand Disaggregation Algorithms
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================


### The numbering for each technique refer to the numbering used in the associated article ("Customer-Level Solar-Demand Disaggregation: The Value of Information").
### Also, for more details on each approach, please refer to the above article. In what follows, we use SDD which stands for solar demand disaggregation


# # ================================================================
# # Technique 1: Minimum Solar Generation
# # ================================================================

def SDD_min_solar_single_node(customer,input_features):

    print(f'customer_ID: {customer.nmi} begin')

    customer.Generate_disaggregation_positive_minimum_PV()

    result = pd.DataFrame(customer.data.pv_disagg)
    result['demand_disagg'] = customer.data.demand_disagg
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return(result)

def SDD_min_solar_mutiple_nodes(customers,input_features):

    predictions_prallel = pool_executor_parallel(SDD_min_solar_single_node,customers.values(),input_features)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return(predictions_prallel)

# # ================================================================
# # Technique 2: Same Irradiance
# # ================================================================

def SDD_Same_Irrad_single_time(time_step,customers_nmi_with_pv,datetimes,data_one_time):

    """
    SDD_Same_Irrad(t,customers_nmi_with_pv,datetimes,data_one_time), where t is the time-step of the disaggregation.
    
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
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]] + model.penalty_p[t,i] - model.penalty_n[t,i] 
    model.cons = Constraint(model.Time,customers_nmi_with_pv,rule=load_balance)

    # # Objective
    def obj_rule(model):
        return sum(model.pv[t] for t in model.Time) + 10000 * sum( sum( model.penalty_p[t,i] + model.penalty_n[t,i] for i in customers_nmi_with_pv ) for t in model.Time)
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    print(" Disaggregating {first}-th time step".format(first = t))
    # print(t)

    result_output_temp =  ({i:    (model.pv[t].value * data_one_time.loc[i].pv_system_size[0] + model.penalty_p[t,i].value)  for i in customers_nmi_with_pv},
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

def SDD_Same_Irrad_for_parallel(time_step,customers_nmi_with_pv,datetimes):

    """
    disaggregate_demand(t,customers_nmi_with_pv,customers), where t is the time-step of the disaggregation.
    
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
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]] + model.penalty_p[t,i] - model.penalty_n[t,i] 
    model.cons = Constraint(model.Time,customers_nmi_with_pv,rule=load_balance)

    # # Objective
    def obj_rule(model):
        return sum(model.pv[t] for t in model.Time) + 10000 * sum( sum( model.penalty_p[t,i] + model.penalty_n[t,i] for i in customers_nmi_with_pv ) for t in model.Time)
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    print(" Disaggregating {first}-th time step".format(first = t))

    result_output_temp =  ({i:    (model.pv[t].value * data_one_time.loc[i].pv_system_size[0] + model.penalty_p[t,i].value)  for i in customers_nmi_with_pv},
            {i:      model.demand[t,i].value + model.penalty_n[t,i].value  for i in customers_nmi_with_pv} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)

    return result_output

def pool_executor_parallel_time(function_name,repeat_iter,customers_nmi_with_pv,datetimes,data,input_features):
    
    global shared_data_disaggregation_optimisation

    shared_data_disaggregation_optimisation = copy(data)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,repeat(customers_nmi_with_pv),repeat(datetimes)))  
    return results

def SDD_Same_Irrad_multiple_times(data,input_features,datetimes,customers_nmi_with_pv):

    """
    Generate_disaggregation_optimisation()
    
    This function disaggregates the demand and generation for all the nodes in the system and all the time-steps, and adds the disaggergations to each
    class variable. It applies the disaggregation to all nmis. This fuction uses function "pool_executor_disaggregation" to run the disaggregation algorithm.  
    """

    global shared_data_disaggregation_optimisation

    predictions_prallel = pool_executor_parallel_time(SDD_Same_Irrad_for_parallel,range(0,len(datetimes)),customers_nmi_with_pv,datetimes,data,input_features)
    
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    print('Done')

    # print(len(predictions_prallel))
    
    if 'shared_data_disaggregation_optimisation' in globals():
        del(shared_data_disaggregation_optimisation)

    return predictions_prallel

# # ================================================================
# # Technique 3: Same Irradiance and Houses Without PV Installation
# # ================================================================
def SDD_Same_Irrad_no_PV_houses_single_time(time_step,data,customers_with_pv,customers_without_pv,datetimes):
    
    t = time_step

    model=ConcreteModel()

    data_one_time = data.loc[pd.IndexSlice[:, datetimes[t]], :]

    model.Time = Set(initialize=range(t,t+1))
    model.pv=Var(model.Time,customers_with_pv, bounds=(0,1))
    model.demand=Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_p=Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_n=Var(model.Time,customers_with_pv,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t,i] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]]
    model.cons = Constraint(model.Time,customers_with_pv,rule=load_balance)

    # # Objective
    def obj_rule(model):
        return (sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]]/len(customers_without_pv) for i in customers_without_pv) 
                + sum(sum(model.pv[t,i]**2 for i in customers_with_pv) for t in model.Time)
                )
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    result_output_temp =  ({i:    (model.pv[t,i].value * data_one_time.loc[i].pv_system_size[0])  for i in customers_with_pv},
            {i:      model.demand[t,i].value  for i in customers_with_pv} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)

    return result_output



def pool_executor_parallel_time_no_PV_houses(function_name,repeat_iter,customers_with_pv,customers_without_pv,datetimes,data,input_features):
    
    global shared_data_disaggregation_optimisation_no_PV

    shared_data_disaggregation_optimisation_no_PV = copy(data)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,repeat(customers_with_pv),repeat(customers_without_pv),repeat(datetimes)))  
    return results


def SDD_Same_Irrad_no_PV_houses_multiple_times(data,input_features,datetimes,customers_with_pv,customers_without_pv):

    """
    Generate_disaggregation_optimisation()
    
    This function disaggregates the demand and generation for all the nodes in the system and all the time-steps, and adds the disaggergations to each
    class variable. It applies the disaggregation to all nmis. This fuction uses function "pool_executor_disaggregation" to run the disaggregation algorithm.  
    """

    global shared_data_disaggregation_optimisation_no_PV

    predictions_prallel = pool_executor_parallel_time_no_PV_houses(SDD_Same_Irrad_no_PV_houses_single_time_for_parallel,range(0,len(datetimes)),customers_with_pv,customers_without_pv,datetimes,data,input_features)
    
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    print('Done')

    # print(len(predictions_prallel))
    
    if 'shared_data_disaggregation_optimisation_no_PV' in globals():
        del(shared_data_disaggregation_optimisation_no_PV)

    return predictions_prallel


def SDD_Same_Irrad_no_PV_houses_single_time_for_parallel(time_step,customers_with_pv,customers_without_pv,datetimes):

    t = time_step

    model=ConcreteModel()

    data_one_time = shared_data_disaggregation_optimisation_no_PV.loc[pd.IndexSlice[:, datetimes[t]], :]

    model.Time = Set(initialize=range(t,t+1))
    model.pv=Var(model.Time,customers_with_pv, bounds=(0,1))
    model.demand=Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_p=Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_n=Var(model.Time,customers_with_pv,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t,i] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]]
    model.cons = Constraint(model.Time,customers_with_pv,rule=load_balance)

    # # Objective
    def obj_rule(model):
        return (sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]]/len(customers_without_pv) for i in customers_without_pv) 
                + sum(sum(model.pv[t,i]**2 for i in customers_with_pv) for t in model.Time)
                )
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    result_output_temp =  ({i:    (model.pv[t,i].value * data_one_time.loc[i].pv_system_size[0])  for i in customers_with_pv},
            {i:      model.demand[t,i].value  for i in customers_with_pv} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)

    print(" Disaggregating {first}-th time step".format(first = t))

    return result_output


# # ================================================================
# # Technique 4: Constant Power Factor Demand
# # ================================================================

def SDD_constant_PF_single_node(customer,input_features):

    customer.generate_disaggregation_using_reactive()

    result = pd.DataFrame(customer.data.pv_disagg)
    result['demand_disagg'] = customer.data.demand_disagg
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return(result)

def SDD_constant_PF_mutiple_nodes(customers,input_features):

    predictions_prallel = pool_executor_parallel(SDD_constant_PF_single_node,customers.values(),input_features)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return(predictions_prallel)


# # ================================================================
# # Technique 5: Measurements from Neighbouring Sites
# # ================================================================
def SDD_known_pvs_single_node(customer,customers_known_pv,datetimes):

    model=ConcreteModel()
    known_pv_nmis = list(customers_known_pv.keys())
    model.pv_cites = Set(initialize=known_pv_nmis)
    model.Time = Set(initialize=range(0,len(datetimes)))
    model.weight = Var(model.pv_cites, bounds=(0,1))

    # # Constraints
    def load_balance(model):
        return sum(model.weight[i] for i in model.pv_cites) == 1 
    model.cons = Constraint(rule=load_balance)

    # Objective
    def obj_rule(model):
        return  sum(
        ( sum(model.weight[i] * customers_known_pv[i].data.pv[datetimes[t]]/customers_known_pv[i].data.pv_system_size[0] for i in model.pv_cites)
                - max(-customer.data.active_power[datetimes[t]],0)/customer.data.pv_system_size[0]
        )**2 for t in model.Time)

    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)
     
    pv_dis = pd.concat([sum(model.weight[i].value * customers_known_pv[i].data.pv/customers_known_pv[i].data.pv_system_size[0] for i in model.pv_cites) * customer.data.pv_system_size[0],
                    -customer.data.active_power]).max(level=0)
    
    load_dis = customer.data.active_power + pv_dis

    return  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})

def SDD_known_pvs_single_node_for_parallel(customer,datetimes):

    print(f'customer_ID: {customer.nmi} begin')

    model=ConcreteModel()
    known_pv_nmis = list(customers_known_pv_shared.keys())
    model.pv_cites = Set(initialize=known_pv_nmis)
    model.Time = Set(initialize=range(0,len(datetimes)))
    model.weight = Var(model.pv_cites, bounds=(0,1))

    # # Constraints
    def load_balance(model):
        return sum(model.weight[i] for i in model.pv_cites) == 1 
    model.cons = Constraint(rule=load_balance)

    # Objective
    def obj_rule(model):
        return  sum(
        ( sum(model.weight[i] * customers_known_pv_shared[i].data.pv[datetimes[t]]/customers_known_pv_shared[i].data.pv_system_size[0] for i in model.pv_cites)
                - max(-customer.data.active_power[datetimes[t]],0)/customer.data.pv_system_size[0]
        )**2 for t in model.Time)

    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)
     
    pv_dis = pd.concat([sum(model.weight[i].value * customers_known_pv_shared[i].data.pv/customers_known_pv_shared[i].data.pv_system_size[0] for i in model.pv_cites) * customer.data.pv_system_size[0],
                    -customer.data.active_power]).max(level=0)
    
    load_dis = customer.data.active_power + pv_dis

    return  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})



def pool_executor_parallel_knownPVS(function_name,repeat_iter,input_features,customers_known_pv,datetimes):
    
    global customers_known_pv_shared

    customers_known_pv_shared = copy(customers_known_pv)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,repeat(datetimes)))  
    return results


def SDD_known_pvs_multiple_nodes(customers,input_features,customers_known_pv,datetimes):


    global customers_known_pv_shared

    predictions_prallel = pool_executor_parallel_knownPVS(SDD_known_pvs_single_node_for_parallel,customers.values(),input_features,customers_known_pv,datetimes)
    
    predictions_output = {list(customers.keys())[i]: predictions_prallel[i] for i in range(0,len(customers.keys()))}

    # predictions_prallel = pd.concat(predictions_prallel, axis=0)

    if 'customers_known_pv_shared' in globals():
        del(customers_known_pv_shared)

    return(predictions_output)



# # ================================================================
# # Technique 6: Weather Data
# # ================================================================
def SDD_using_temp_single_node(customer,data_weather):
    weather = copy(data_weather['Loc1'])
    weather['minute'] = weather.index.minute
    weather['hour'] = weather.index.hour
    weather['isweekend'] = (weather.index.day_of_week > 4).astype(int)
    weather['Temp_EWMA'] = weather.AirTemp.ewm(com=0.5).mean()
    weather = copy(weather[weather.index < '2019-01-01'])
    weather_input = weather[['AirTemp','hour','minute','Temp_EWMA','isweekend']]
    weather_input.set_index(weather_input.index.tz_localize(None),inplace=True)
    weather_input = weather_input[~weather_input.index.duplicated(keep='first')]

    pv_dis = copy(customer.data.active_power)
    pv_dis[pv_dis > 0 ] = 0 
    pv_dis = -pv_dis
    set_diff = list( set(weather_input.index)-set( pv_dis.index) )
    weather_input = weather_input.drop(set_diff)

    load_dis = customer.data.active_power + pv_dis

    iteration = 0
    pv_dis_iter = copy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 10:

        iteration += 1
        pv_dis_iter = copy(pv_dis)
        print(f'Iteration: {iteration}')

        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=customer.data.index)
        load_dis[load_dis < 0 ] = 0 
        pv_dis = load_dis - customer.data.active_power

    pv_dis[pv_dis < 0 ] = 0 
    load_dis =  customer.data.active_power + pv_dis

    return pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})

def SDD_using_temp_single_node_for_parallel(customer):

    print(f'customer_ID: {customer.nmi}')

    weather = copy(shared_weather_data['Loc1'])
    weather['minute'] = weather.index.minute
    weather['hour'] = weather.index.hour
    weather['isweekend'] = (weather.index.day_of_week > 4).astype(int)
    weather['Temp_EWMA'] = weather.AirTemp.ewm(com=0.5).mean()
    weather = copy(weather[weather.index < '2019-01-01'])
    weather_input = weather[['AirTemp','hour','minute','Temp_EWMA','isweekend']]
    weather_input.set_index(weather_input.index.tz_localize(None),inplace=True)
    weather_input = weather_input[~weather_input.index.duplicated(keep='first')]

    pv_dis = copy(customer.data.active_power)
    pv_dis[pv_dis > 0 ] = 0 
    pv_dis = -pv_dis
    set_diff = list( set(weather_input.index)-set( pv_dis.index) )
    weather_input = weather_input.drop(set_diff)

    load_dis = customer.data.active_power + pv_dis

    iteration = 0
    pv_dis_iter = copy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 10:

        iteration += 1
        pv_dis_iter = copy(pv_dis)

        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=customer.data.index)
        load_dis[load_dis < 0 ] = 0 
        pv_dis = load_dis - customer.data.active_power

    pv_dis[pv_dis < 0 ] = 0 
    load_dis =  customer.data.active_power + pv_dis

    return pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})

def pool_executor_parallel_temperature(function_name,repeat_iter,input_features,data_weather):
    
    global shared_weather_data

    shared_weather_data = copy(data_weather)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter))  
    return results

def SDD_using_temp_multilple_nodes(customers,input_features,data_weather):

    global shared_weather_data

    predictions_prallel = pool_executor_parallel_temperature(SDD_using_temp_single_node_for_parallel,customers.values(),input_features,data_weather)
    
    predictions_output = {list(customers.keys())[i]: predictions_prallel[i] for i in range(0,len(customers.keys()))}

    # predictions_prallel = pd.concat(predictions_prallel, axis=0)

    if 'shared_weather_data' in globals():
        del(shared_weather_data)


    return(predictions_output)


# # ================================================================
# # Technique 7: Proxy Measurements from Neighbouring Sites and Weather Data
# # ================================================================
def SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,pv_iter):
    known_pv_nmis = list(customers_known_pv.keys())
    model=ConcreteModel()
    model.pv_cites = Set(initialize=known_pv_nmis)
    model.Time = Set(initialize=range(0,len(datetimes)))
    model.weight=Var(model.pv_cites, bounds=(0,1))

    # # Constraints
    def load_balance(model):
        return sum(model.weight[i] for i in model.pv_cites) == 1 
    model.cons = Constraint(rule=load_balance)

    # Objective
    def obj_rule(model):
        return  sum(
                    (sum(model.weight[i] * customers_known_pv[i].data.pv[datetimes[t]]/customers_known_pv[i].data.pv_system_size[0] for i in model.pv_cites)
                        - pv_iter[datetimes[t]]/customer.data.pv_system_size[0] )**2 for t in model.Time)

    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)
    
    return pd.concat([sum(model.weight[i].value * customers_known_pv[i].data.pv/customers_known_pv[i].data.pv_system_size[0] for i in model.pv_cites) * customer.data.pv_system_size[0],
                    -customer.data.active_power]).max(level=0)

def SDD_known_pvs_temp_single_node_algorithm(customer,data_weather,customers_known_pv,datetimes):
    
    weather = copy(data_weather['Loc1'])
    weather['minute'] = weather.index.minute
    weather['hour'] = weather.index.hour
    weather['isweekend'] = (weather.index.day_of_week > 4).astype(int)
    weather['Temp_EWMA'] = weather.AirTemp.ewm(com=0.5).mean()
    weather = copy(weather[weather.index < '2019-01-01'])
    weather_input = weather[['AirTemp','hour','minute','Temp_EWMA','isweekend']]
    weather_input.set_index(weather_input.index.tz_localize(None),inplace=True)
    weather_input = weather_input[~weather_input.index.duplicated(keep='first')]


    pv_iter0 = copy(customer.data.active_power)
    pv_iter0[pv_iter0 > 0 ] = 0 
    pv_iter0 = -pv_iter0
    set_diff = list( set(weather_input.index)-set( pv_iter0.index) )
    weather_input = weather_input.drop(set_diff)


    pv_dis = SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,pv_iter0)
    load_dis = customer.data.active_power + pv_dis

    iteration = 0
    pv_dis_iter = copy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 10:
        
        iteration += 1
        pv_dis_iter = copy(pv_dis)
        print(f'Iteration: {iteration}')
        
        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=customer.data.index)
        pv_dis = SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,load_dis - customer.data.active_power)
        load_dis = customer.data.active_power + pv_dis

    return pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})


def SDD_known_pvs_temp_single_node_algorithm_for_parallel(customer,datetimes):
    
    weather = copy(shared_weather_data['Loc1'])
    weather['minute'] = weather.index.minute
    weather['hour'] = weather.index.hour
    weather['isweekend'] = (weather.index.day_of_week > 4).astype(int)
    weather['Temp_EWMA'] = weather.AirTemp.ewm(com=0.5).mean()
    weather = copy(weather[weather.index < '2019-01-01'])
    weather_input = weather[['AirTemp','hour','minute','Temp_EWMA','isweekend']]
    weather_input.set_index(weather_input.index.tz_localize(None),inplace=True)
    weather_input = weather_input[~weather_input.index.duplicated(keep='first')]

    pv_iter0 = copy(customer.data.active_power)
    pv_iter0[pv_iter0 > 0 ] = 0 
    pv_iter0 = -pv_iter0
    set_diff = list( set(weather_input.index)-set( pv_iter0.index) )
    weather_input = weather_input.drop(set_diff)


    pv_dis = SDD_known_pvs_temp_single_node(customer,shared_data_known_pv,datetimes,pv_iter0)
    
    print(f'customer_ID: {customer.nmi} begin')
    load_dis = customer.data.active_power + pv_dis

    iteration = 0
    pv_dis_iter = copy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 10:
        
        iteration += 1
        pv_dis_iter = copy(pv_dis)
        # print(iteration)
        
        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=customer.data.index)
        pv_dis = SDD_known_pvs_temp_single_node(customer,shared_data_known_pv,datetimes,load_dis - customer.data.active_power)
        load_dis = customer.data.active_power + pv_dis

    print(f'customer_ID: {customer.nmi} done!')

    return pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})


def pool_executor_parallel_known_pvs_temp(function_name,repeat_iter,input_features,data_weather,customers_known_pv,datetimes):
    
    global shared_data_known_pv
    global shared_weather_data

    shared_data_known_pv = copy(customers_known_pv)
    shared_weather_data = copy(data_weather)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,repeat(datetimes)))  
    return results


def SDD_known_pvs_temp_multiple_node_algorithm(customers,input_features,data_weather,customers_known_pv,datetimes):

    global shared_data_known_pv
    global shared_weather_data

    predictions_prallel = pool_executor_parallel_known_pvs_temp(SDD_known_pvs_temp_single_node_algorithm_for_parallel,customers.values(),input_features,data_weather,customers_known_pv,datetimes)
    
    predictions_output = {list(customers.keys())[i]: predictions_prallel[i] for i in range(0,len(customers.keys()))}

    # predictions_prallel = pd.concat(predictions_prallel, axis=0)

    if 'shared_data_known_pv' in globals():
        del(shared_data_known_pv)
    if 'shared_weather_data' in globals():
        del(shared_weather_data)

    return(predictions_output)






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

# # from read_data_init import input_features
# [data, customers_nmi,customers_nmi_with_pv,datetimes, customers] = read_data(input_features)






# # ================================================================
# # To be tested techniques
# # ================================================================


### To be tested: This function uses the linear regression model which is built-in the sklearn librarry to find a linear relation between the PV system size and solar generation at each time step for all the nodes.
### I think it should work almost exactly the same as technique 2 as it uses the same assumptions. But it requires more testing to be sure.
# def generate_disaggregation_regression(customers,customers_nmi,customers_nmi_with_pv,datetimes):

#     """
#     generate_disaggregation_regression(customers)
    
#     This function uses a linear regression model to disaggregate the electricity demand from PV generation in the data. 
#     Note that the active power stored in the data is recorded at at each \textit{nmi} connection point to the grid and thus 
#     is summation all electricity usage and generation that happens behind the meter. 
#     """

#     pv = [ Ridge(alpha=1.0).fit( np.array([customers[i].data.pv_system_size[datetimes[0]]/customers[i].data.active_power.max()  for i in customers_nmi_with_pv]).reshape((-1,1)),
#                             np.array([customers[i].data.active_power[datetimes[t]]/customers[i].data.active_power.max() for i in customers_nmi_with_pv])     
#                             ).coef_[0]   for t in range(0,len(datetimes))]

#     for i in customers_nmi_with_pv:
#         customers[i].data['pv_disagg'] = [ pv[t]*customers[i].data.pv_system_size[datetimes[0]] + min(customers[i].data.active_power[datetimes[t]] + pv[t]*customers[i].data.pv_system_size[datetimes[0]],0) for t in range(0,len(datetimes))]
#         customers[i].data['demand_disagg'] = [max(customers[i].data.active_power[datetimes[t]] + pv[t]*customers[i].data.pv_system_size[datetimes[0]],0) for t in range(0,len(datetimes))]

#     for i in list(set(customers_nmi) - set(customers_nmi_with_pv)):
#         customers[i].data['pv_disagg'] = 0
#         customers[i].data['demand_disagg'] = customers[customers_nmi[0]].data.active_power.values


### To be tested 
# def Forecast_using_disaggregation(data,input_features,datetimes,customers_nmi_with_pv,customers_nmi,customers):

#     """
#     Forecast_using_disaggregation(customers_nmi,input_features)

#     This function is used to generate forecast values. It first disaggregates the demand and generation for all nmi using function
#     Generate_disaggregation_optimisation (technique 2). It then uses function forecast_pointbased for the disaggregated demand and generation and produces separate forecast. It finally sums up the two values
#     and returns an aggregated forecast for all nmis in pandas.Dataframe format.
#     """
    
#     opt_disaggregate = disaggregation_optimisation(data,input_features,datetimes,customers_nmi_with_pv)

#     for i in customers_nmi_with_pv:
#         customers[i].data['pv_disagg'] = opt_disaggregate.loc[i].pv_disagg.values
#         customers[i].data['demand_disagg'] = opt_disaggregate.loc[i].pv_disagg.values

#     for i in list(set(customers_nmi) - set(customers_nmi_with_pv)):
#         customers[i].data['pv_disagg'] = 0
#         customers[i].data['demand_disagg'] = customers[i].data.active_power.values


#     input_features_copy = copy(input_features)
#     input_features_copy['Forecasted_param']= 'pv_disagg'
#     predictions_output_pv = forecast_pointbased(customers,input_features_copy)

#     input_features_copy['Forecasted_param']= 'demand_disagg'
#     predictions_output_demand = forecast_pointbased(customers,input_features_copy)

#     predictions_agg = predictions_output_demand + predictions_output_pv

#     return(predictions_agg)



#### Approach to be tested
# def disaggregation_single_known_pvs_pos_pv(nmi,known_pv_nmis,customers,datetimes):

#     model=ConcreteModel()
#     model.pv_cites = Set(initialize=known_pv_nmis)
#     model.Time = Set(initialize=range(0,len(datetimes)))
#     model.weight=Var(model.pv_cites, bounds=(0,1))
#     model.irrid=Var(range(0,len(datetimes)),bounds=(0,1))

#     # # Constraints
#     def load_balance(model):
#         return sum(model.weight[i] for i in model.pv_cites) == 1 
#     model.cons = Constraint(rule=load_balance)

#     def pv_irrid(model,t):
#         return model.irrid[t] == sum(model.weight[i] * customers[i].data.pv[datetimes[t]]/customers[i].data.pv_system_size[0] for i in model.pv_cites)
#     model.cons_pv = Constraint(model.Time,rule=pv_irrid)

#     # Objective
#     def obj_rule(model):
#         return  sum( (model.irrid[t] - max(-customers[nmi].data.active_power[datetimes[t]],0)/customers[nmi].data.pv_system_size[0] )**2
#                       for t in model.Time)

#     model.obj=Objective(rule=obj_rule)

#     # # Solve the model
#     opt = SolverFactory('gurobi')
#     opt.solve(model)

#     return pd.Series([model.irrid[t].value * customers[nmi].data.pv_system_size[0] for t in model.Time],index=customers[nmi].data.index)
