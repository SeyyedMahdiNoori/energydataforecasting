from operator import le
from sqlite3 import Time
import pandas as pd
import numpy as np
import pickle
import copy
from sklearn.linear_model import Ridge,LinearRegression # Linear least squares with l2 regularization. (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
from sklearn.pipeline import make_pipeline # Construct a Pipeline from the given estimators (for more information: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance.
from sklearn.ensemble import RandomForestRegressor

from skforecast.ForecasterAutoreg import ForecasterAutoreg # A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
from skforecast.model_selection import grid_search_forecaster, backtesting_forecaster
from datetime import date, timedelta, datetime
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
import itertools
import connectorx as cx
from tsprial.forecasting import ForecastingChain
from tsprial.forecasting import ForecastingStacked
from sklearn.tree import DecisionTreeRegressor
from tsprial.forecasting import ForecastingRectified
from dateutil import parser
from dateutil.tz import tzutc

# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# A function to decide whether a string in the form of datetime has a time zone or not
def has_timezone(string):
    '''
    has_timezone(string) accept string in the form of datetime and return True if it has timezone, and it returns False otherwise.
    '''
    try:
        parsed_date = parser.parse(string)
        return parsed_date.tzinfo is not None
    except (TypeError, ValueError):
        return False


def initialise(customersdatapath=None,raw_data=None,forecasted_param=None,weatherdatapath=None,raw_weather_data=None,start_training=None,end_training=None,nmi_type_path=None,Last_observed_window=None,window_size=None,windows_to_be_forecasted=None,core_usage=None,db_url=None,db_table_names=None):
    '''
    initialise(customersdatapath=None,raw_data=None,forecasted_param=None,weatherdatapath=None,raw_weather_data=None,start_training=None,end_training=None,nmi_type_path=None,Last_observed_window=None,window_size=None,windows_to_be_forecasted=None,core_usage=None,db_url=None,db_table_names=None)

    This function is to initialise the data and the input parameters required for the rest of the functions in this package. It requires one of the followings: 
    1. a path to a csv file 2. raw_data or 3. database url and the associate table names in that url. Other inputs are all optional.  
    '''
    # Read data
    if customersdatapath is not None:
        data = pd.read_csv(customersdatapath)     
    elif raw_data is not None:
        data = copy(raw_data)
    elif db_url is not None and db_table_names is not None:
        sql = [f"SELECT * from {table}" for table in db_table_names]
        data = cx.read_sql(db_url,sql)
        data.sort_values(by='datetime',inplace=True)
    else:
        return f'Error!!! Either customersdatapath, raw_data or db_url needs to be provided'
    

    # # ###### Pre-process the data ######
    # format datetime to pandas datetime format
    check_time_zone = has_timezone(data.datetime[0])
    if check_time_zone == False:
        data['datetime'] = pd.to_datetime(data['datetime'])
    else:
        data['datetime'] = pd.to_datetime(data['datetime'], utc=True, infer_datetime_format=True)
        data["datetime"] = data["datetime"].dt.tz_convert("Australia/Sydney")

    # Add weekday column to the data
    # data['DayofWeek'] = data['datetime'].dt.day_name()

    # Save customer nmis in a list
    customers_nmi = list(dict.fromkeys(data['nmi'].values.tolist()))

    # Make datetime index of the dataset
    data.set_index(['nmi', 'datetime'], inplace=True)

    # save unique dates of the data
    datetimes = data.index.unique('datetime')

    # read and process weather data if it has been inputted
    if weatherdatapath is None and raw_weather_data is None:
        data_weather = {}
    elif weatherdatapath is not None:
        data_weather = pd.read_csv(weatherdatapath)
        data_weather['PeriodStart'] = pd.to_datetime(data_weather['PeriodStart'], utc=True, infer_datetime_format=True)
        data_weather = data_weather.drop('PeriodEnd', axis=1)
        data_weather = data_weather.rename(columns={"PeriodStart": "datetime"})
        data_weather.set_index('datetime', inplace=True)
        data_weather.index = data_weather.index.tz_convert('Australia/Sydney')

        data_weather['minute'] = data_weather.index.minute
        data_weather['hour'] = data_weather.index.hour
        data_weather['isweekend'] = (data_weather.index.day_of_week > 4).astype(int)
        data_weather['Temp_EWMA'] = data_weather.AirTemp.ewm(com=0.5).mean()
        data_weather.set_index(data_weather.index.tz_localize(None),inplace=True)
        data_weather = data_weather[~data_weather.index.duplicated(keep='first')]

        # remove rows that have a different index from datetimes (main data index). This keeps them with the same lenght later on when the 
        # weather data is going to be used for learning
        set_diff = list( set(data_weather.index)-set( datetimes) )
        data_weather = data_weather.drop(set_diff)
        
        # fill empty rows (rows that are in the main data and not available in the weather data) with average over the same day.
        set_diff = list( set( datetimes) - set(data_weather.index) )
        for i in range(0,len(set_diff)):
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().AirTemp,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().Temp_EWMA,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                    ],ignore_index=False)
            # data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':17.5,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':17.5,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
            #                         ],ignore_index=False)

    else:
        data_weather = copy(raw_weather_data)
        data_weather['PeriodStart'] = pd.to_datetime(data_weather['PeriodStart'])
        data_weather = data_weather.drop('PeriodEnd', axis=1)
        data_weather = data_weather.rename(columns={"PeriodStart": "datetime"})
        data_weather.set_index('datetime', inplace=True)
        data_weather.index = data_weather.index.tz_convert('Australia/Sydney')

        data_weather['minute'] = data_weather.index.minute
        data_weather['hour'] = data_weather.index.hour
        data_weather['isweekend'] = (data_weather.index.day_of_week > 4).astype(int)
        data_weather['Temp_EWMA'] = data_weather.AirTemp.ewm(com=0.5).mean()
        data_weather.set_index(data_weather.index.tz_localize(None),inplace=True)
        data_weather = data_weather[~data_weather.index.duplicated(keep='first')]

        set_diff = list( set(data_weather.index)-set( datetimes) )
        data_weather = data_weather.drop(set_diff)

        set_diff = list( set( datetimes) - set(data_weather.index) )
        for i in range(0,len(set_diff)):
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().AirTemp,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().Temp_EWMA,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                    ],ignore_index=False)

    # create and populate input_features which is a paramter that will be used in almost all the functions in this package.
    # This paramtere represent the input preferenes. If there is no input to the initial() function to fill this parameters,
    # defeault values will be used to fill in the gap. 
    input_features = {}

    # The parameters to be forecasted. It should be a column name in the input data.
    if forecasted_param is None:
        input_features['Forecasted_param'] = 'active_power'
    else:
        input_features['Forecasted_param'] = forecasted_param

    # The datetime index that training starts from
    if start_training is None:
        input_features['Start training'] = datetimes[0].strftime("%Y-%m-%d %H:%M:%S")
    else:
        input_features['Start training'] = start_training + ' ' + '00:00:00'

    # The last datetime index used for trainning.
    if end_training is None:
        input_features['End training'] = (datetimes[-1] - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    else:
        input_features['End training'] = end_training + ' ' + '00:00:00'

    # Select customers with pv. This is used to read the "nmi.csv" file, used in the Converge project.
    if nmi_type_path is None:
        customers_nmi_with_pv = copy(customers_nmi)
    else:
        input_features['nmi_type_path'] = nmi_type_path
        data_nmi = pd.read_csv(nmi_type_path)
        data_nmi.set_index(data_nmi['nmi'],inplace=True)

        customers_nmi_with_pv = [ data_nmi.loc[i]['nmi'] for i in customers_nmi if data_nmi.loc[i]['has_pv']==True ]
        data['has_pv']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['has_pv']] for i in customers_nmi]* len(datetimes)))
        data['customer_kind']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['customer_kind']] for i in customers_nmi]* len(datetimes)))
        data['pv_system_size']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['pv_system_size']] for i in customers_nmi]* len(datetimes)))

    # The last obersved window. The forecasting values are generated after this time index.
    if Last_observed_window is None:
        input_features['Last-observed-window'] = input_features['End training']
    else:
        input_features['Last-observed-window'] = Last_observed_window + ' ' + '00:00:00'

    # Size of each window to be forecasted. A window is considered to be a day, and the resolution of the data is considered as the window size.
    # For example, for a data with resolution 30th minutely, the window size woul be 48.
    if window_size is None:
        input_features['Window size'] = int(timedelta(days = 1) / (datetimes[1] - datetimes[0]))
    else:
        input_features['Window size'] = window_size

    # The number of days to be forecasted.
    if windows_to_be_forecasted is None:
        input_features['Windows to be forecasted'] = 1
    else:
        input_features['Windows to be forecasted'] = windows_to_be_forecasted

    # Data forequency.
    input_features['data_freq'] = datetimes[0:3].inferred_freq

    # number of processes parallel programming.
    if core_usage is None:
        input_features['core_usage'] = 8
    else:
        input_features['core_usage'] = core_usage

    if check_time_zone == True:
        datetimes.freq = input_features['data_freq']
        data.index.levels[1].freq = input_features['data_freq']

    # Customers is a class. An instant is assigned to each customer with its data. Most of the forecasting functions are then defined 
    # as methods within this class.
    global Customers

    class Customers:
        
        check_time_zone_class = check_time_zone
        
        num_of_customers = 0

        def __init__(self, nmi,input_features):

            self.nmi = nmi      # store nmi in each object              
            self.data = data.loc[self.nmi]      # store data in each object         

            Customers.num_of_customers += 1

        def generate_forecaster_autoregressive(self,input_features):            
            """
            generate_forecaster_autoregressive(input_features)
            
            This function generates a forecaster object to be used for a autoregressive recursive multi-step forecasting method. 
            It is based on a linear least squares with l2 regularization method. Alternatively, LinearRegression() and Lasso() that
            have different objective can be used with the same parameters.
            """

            # Create a forecasting object
            self.forecaster_autoregressive = ForecasterAutoreg(
                    regressor = make_pipeline(StandardScaler(), Ridge()),  
                    lags      = input_features['Window size']      
                )

            # Train the forecaster using the train data
            self.forecaster_autoregressive.fit(y=self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']])

        def generate_forecaster_direct(self,input_features):            
            """
            generate_forecaster_direct(self,input_features)
            
            This function generates a forecaster object to be used for a multi-step forecasting method. 
            More details about this approach can be found in "https://robjhyndman.com/papers/rectify.pdf" and "https://towardsdatascience.com/6-methods-for-multi-step-forecasting-823cbde4127a"
            """

            self.forecaster_direct = ForecastingChain(
                        Ridge(),
                        n_estimators=input_features['Window size'],
                        lags=range(1,input_features['Window size']+1),
                        use_exog=False,
                        accept_nan=False
                                            )
            self.forecaster_direct.fit(None, self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']])

        def generate_forecaster_stacking(self,input_features):            
            """
            generate_forecaster_stacking(self,input_features)
            
            This function generates a forecaster object to be used for a multi-step forecasting method. 
            More details about this approach can be found in "https://towardsdatascience.com/6-methods-for-multi-step-forecasting-823cbde4127a"
            """

            self.forecaster_stacking = ForecastingStacked(
                        [Ridge(), DecisionTreeRegressor()],
                        test_size = input_features['Window size']* input_features['Windows to be forecasted'],
                        lags=range(1,input_features['Window size']+1),
                        use_exog=False
                                            )
            self.forecaster_stacking.fit(None, self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']])


        def generate_forecaster_rectified(self,input_features):            
            """
            generate_forecaster_rectified(self,input_features)
            
            This function generates a forecaster object to be used for a multi-step forecasting method. 
            More details about this approach can be found in "https://robjhyndman.com/papers/rectify.pdf" and "https://towardsdatascience.com/6-methods-for-multi-step-forecasting-823cbde4127a"
            """

            self.forecaster_rectified = ForecastingRectified(
                        Ridge(),
                        n_estimators=200,
                        test_size = input_features['Window size']* input_features['Windows to be forecasted'],
                        lags=range(1,input_features['Window size']+1),
                        use_exog=False
                                            )
            self.forecaster_rectified.fit(None, self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']])


        def generate_optimised_forecaster_object(self,input_features):            
            """
            generate_optimised_forecaster_object(self,input_features)
            
            This function generates a forecaster object for each \textit{nmi} to be used for a recursive multi-step forecasting method.
            It builds on function Generate\_forecaster\_object by combining grid search strategy with backtesting to identify the combination of lags 
            and hyperparameters that achieve the best prediction performance. As default, it is based on a linear least squares with \textit{l2} regularisation method. 
            Alternatively, it can use LinearRegression() and Lasso() methods to generate the forecaster object.
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
            lags_grid = [list(range(1,24)), list(range(1,48)), list(range(1,72))]

            # optimise the forecaster
            grid_search_forecaster(
                            forecaster  = self.forecaster,
                            y           = self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']],
                            param_grid  = param_grid,
                            # lags_grid   = lags_grid,
                            steps       =  input_features['Window size'],
                            metric      = 'mean_absolute_error',
                            # refit       = False,
                            initial_train_size = len(self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']]) - input_features['Window size'] * input_features['Windows to be forecasted'],
                            # fixed_train_size   = False,
                            return_best = True,
                            verbose     = False
                    )
            

        def generate_prediction_autoregressive(self,input_features):
            """
            generate_prediction_autoregressive(self,input_features)
            
            This function outputs the prediction values using a Recursive autoregressive multi-step point-forecasting method.          
            """
            
            # generate datetime index for the predicted values based on the window size and the last obeserved window.
            if self.check_time_zone_class == True:
                new_index =  pd.date_range(
                                            start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                            end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                            freq=input_features['data_freq'],
                                            tz="Australia/Sydney").delete(-1)
            else:
                new_index =  pd.date_range(
                                start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                freq=input_features['data_freq']).delete(-1)

            self.predictions_autoregressive = self.forecaster_autoregressive.predict(steps=len(new_index), last_window=self.data[input_features['Forecasted_param']].loc[(datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']]).to_frame().set_index(new_index)
            
        
        def generate_prediction_direct(self,input_features):
            """
            generate_prediction(self,input_features)
            
            This function outputs the prediction values using a Recursive direct multi-step point-forecasting method.  
            """
            
            # generate datetime index for the predicted values based on the window size and the last obeserved window.
            if self.check_time_zone_class == True:
                new_index =  pd.date_range(
                                            start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                            end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                            freq=input_features['data_freq'],
                                            tz="Australia/Sydney").delete(-1)
            else:
                new_index =  pd.date_range(
                                start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                freq=input_features['data_freq']).delete(-1) 

            self.predictions_direct = pd.DataFrame(self.forecaster_direct.predict(np.arange(len(new_index))),index=new_index,columns=['pred'])

        def generate_prediction_stacking(self,input_features):
            """
            generate_prediction(self,input_features)
            
            This function outputs the prediction values using a Recursive stacking multi-step point-forecasting method. 
            """
            
            # generate datetime index for the predicted values based on the window size and the last obeserved window.
            if self.check_time_zone_class == True:
                new_index =  pd.date_range(
                                            start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                            end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                            freq=input_features['data_freq'],
                                            tz="Australia/Sydney").delete(-1)
            else:
                new_index =  pd.date_range(
                                start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                freq=input_features['data_freq']).delete(-1)            
            
            self.predictions_stacking = pd.DataFrame(self.forecaster_stacking.predict(np.arange(len(new_index))),index=new_index,columns=['pred'])


        def generate_prediction_rectified(self,input_features):
            """
            generate_prediction(self,input_features)
            
            This function outputs the prediction values using a Recursive rectified multi-step point-forecasting method. 
            """
            
            # generate datetime index for the predicted values based on the window size and the last obeserved window.
            if self.check_time_zone_class == True:
                new_index =  pd.date_range(
                                            start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                            end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                            freq=input_features['data_freq'],
                                            tz="Australia/Sydney").delete(-1)
            else:
                new_index =  pd.date_range(
                                start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                freq=input_features['data_freq']).delete(-1) 

            self.predictions_rectified = pd.DataFrame(self.forecaster_rectified.predict(np.arange(len(new_index))),index=new_index,columns=['pred'])

        def generate_interval_prediction(self,input_features):
            """
            generate_interval_prediction(self,input_features)
            
            This function outputs three sets of values (a lower bound, an upper bound and the most likely value), using a recursive multi-step probabilistic forecasting method.
            The confidence level can be set in the function parameters as "interval = [10, 90]".
            """

            # generate datetime index for the predicted values based on the window size and the last obeserved window.
            if self.check_time_zone_class == True:
                new_index =  pd.date_range(
                                            start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                            end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                            freq=input_features['data_freq'],
                                            tz="Australia/Sydney").delete(-1)
            else:
                new_index =  pd.date_range(
                                start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                                end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                                freq=input_features['data_freq']).delete(-1)       

            # [10 90] considers 80% (90-10) confidence interval ------- n_boot: Number of bootstrapping iterations used to estimate prediction intervals.
            self.interval_predictions = self.forecaster_autoregressive.predict_interval(steps=len(new_index),
                                                                         interval = [10, 90],
                                                                         n_boot = 1000,
                                                                         last_window = self.data[input_features['Forecasted_param']].loc[(datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']]
                                                                         ).set_index(new_index)
            # self.predictions                   = self.forecaster.predict(steps=len(new_index),                                    last_window=self.data[input_features['Forecasted_param']].loc[(datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']]).to_frame().set_index(new_index)


        def generate_disaggregation_using_reactive(self):
            '''
            generate_disaggregation_using_reactive()

            Dissaggregate the solar generation and demand from the net real power measurement at the connection point.
            This approach uses reactive power as an indiction. More about this approach can be found in "Customer-Level Solar-Demand Disaggregation: The Value of Information".
            '''
            
            QP_coeff = (self.data.reactive_power.between_time('0:00','5:00')/self.data.active_power.between_time('0:00','5:00')[self.data.active_power.between_time('0:00','5:00') > 0.001]).resample('D').mean()
            QP_coeff[pd.Timestamp((QP_coeff.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d"))] = QP_coeff[-1]
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
            '''
            generate_disaggregation_using_reactive()
            
            Dissaggregate the solar generation and demand from the net real power measurement at the connection point.
            This approach uses the negative and possitive values in the net measurement and assumes the minimum possible PV generation values.
            More about this approach can be found in "Customer-Level Solar-Demand Disaggregation: The Value of Information".
            '''

            D = copy(self.data.active_power)
            D[D<=0] = 0
            S = copy(self.data.active_power)
            S[S>=0] = 0
            self.data['pv_disagg'] =  - S
            self.data['demand_disagg'] = D

    # A dictionary of all the customers with keys being customers_nmi and values being their associated Customers (which is a class) instance.
    customers = {customer: Customers(customer,input_features) for customer in customers_nmi}

    return data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather, input_features


# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Solar and Demand Forecasting functions
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================

def pool_executor_parallel(function_name,repeat_iter,input_features):
    '''
    pool_executor_parallel(function_name,repeat_iter,input_features)
    
    This function is used to parallelised the forecasting for each nmi
    '''
    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,repeat(input_features)))  
    return results


# # ================================================================
# # Autoregressive recursive multi-step point-forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_pointbased_autoregressive_single_node(customer,input_features):
    """
    forecast_pointbased_autoregressive_single_node(customers_nmi,input_features)

    This function generates forecasting values for each customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is the customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    
    # print(customer's nmi)
    print(" Customer nmi: {first}".format(first = customer.nmi))

    # Train a forecasting object
    customer.generate_forecaster_autoregressive(input_features)
    
    # Generate predictions 
    customer.generate_prediction_autoregressive(input_features)

    result = customer.predictions_autoregressive
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


def forecast_pointbased_autoregressive_multiple_nodes(customers,input_features):
    """
    forecast_pointbased_autoregressive_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    predictions_prallel = pool_executor_parallel(forecast_pointbased_autoregressive_single_node,customers.values(),input_features)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel


# # ================================================================
# # Recitifed recursive multi-step point-forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_pointbased_rectified_single_node(customer,input_features):
    """
    forecast_pointbased_rectified_single_node(customers_nmi,input_features)

    This function generates forecasting values for each customer using the rectified recursive multi-step forecasting method.
    It requires two inputs. The first input is the customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    
    # print(customers)
    print(" Customer nmi: {first}".format(first = customer.nmi))

    # Train a forecasting object
    customer.generate_forecaster_rectified(input_features)
    
    # Generate predictions 
    customer.generate_prediction_rectified(input_features)

    result = customer.predictions_rectified
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


def forecast_pointbased_rectified_multiple_nodes(customers,input_features):
    """
    forecast_pointbased_rectified_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the rectified recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    predictions_prallel = pool_executor_parallel(forecast_pointbased_rectified_single_node,customers.values(),input_features)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel


# # ================================================================
# # Direct recursive multi-step point-forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_pointbased_direct_single_node(customer,input_features):
    """
    forecast_pointbased_direct_single_node(customers_nmi,input_features)

    This function generates forecasting values for each customer using the Direct recursive multi-step forecasting method.
    It requires two inputs. The first input is the customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    
    # print(customers)
    print(" Customer nmi: {first}".format(first = customer.nmi))

    # Train a forecasting object
    customer.generate_forecaster_direct(input_features)
    
    # Generate predictions 
    customer.generate_prediction_direct(input_features)

    result = customer.predictions_direct
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


def forecast_pointbased_direct_multiple_nodes(customers,input_features):
    """
    forecast_pointbased_direct_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the Direct recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    predictions_prallel = pool_executor_parallel(forecast_pointbased_direct_single_node,customers.values(),input_features)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel

# # ================================================================
# # Stacking recursive multi-step point-forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_pointbased_stacking_single_node(customer,input_features):
    """
    forecast_pointbased_stacking_single_node(customers_nmi,input_features)

    This function generates forecasting values for each customer using the Stacking recursive multi-step forecasting method.
    It requires two inputs. The first input is the customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    
    # print(customers)
    print(" Customer nmi: {first}".format(first = customer.nmi))

    # Train a forecasting object
    customer.generate_forecaster_stacking(input_features)
    
    # Generate predictions 
    customer.generate_prediction_stacking(input_features)

    result = customer.predictions_stacking
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


def forecast_pointbased_stacking_multiple_nodes(customers,input_features):
    """
    forecast_pointbased_stacking_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the Stacking recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    predictions_prallel = pool_executor_parallel(forecast_pointbased_stacking_single_node,customers.values(),input_features)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel


# # ================================================================
# # Recursive multi-step probabilistic forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_inetervalbased_single_node(customer,input_features):
    """
    forecast_inetervalbased_single_node(customers_nmi,input_features)

    This function generates forecasting values for each customer using the interval-based recursive multi-step forecasting method.
    It requires two inputs. The first input is the customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    print(" Customer nmi: {first}".format(first = customer.nmi))


    # Train a forecasting object
    customer.generate_forecaster_autoregressive(input_features)
    
    # Generate interval predictions 
    customer.generate_interval_prediction(input_features)
    
    result = customer.interval_predictions
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result

def forecast_inetervalbased_multiple_nodes(customers,input_features):
    """
    forecast_inetervalbased_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the Interval-based recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    predictions_prallel = pool_executor_parallel(forecast_inetervalbased_single_node,customers.values(),input_features)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel

# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meters
# # ================================================================
def forecast_lin_reg_proxy_measures_single_node(hist_data_proxy_customers,customer,input_features):
    """
    forecast_lin_reg_proxy_measures(hist_data_proxy_customers,customer,input_features)

    This function generates forecasting values a customer using the some customers real-time measurements as proxies for a target customer.
    It generates a linear function mapping the real-time measurements from know customers to the target customer values.

    It requires three inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    And, the third input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """

    # Create a LinearRegression function from the historical data of proxy and target customers
    reg = LinearRegression().fit(
                                np.transpose(np.array(
                                    [hist_data_proxy_customers[i].data[input_features['Forecasted_param']][input_features['Start training']:input_features['End training']].tolist() for i in hist_data_proxy_customers.keys()]
                                                )       ),
                                    np.array(customer.data[input_features['Forecasted_param']][input_features['Start training']:input_features['End training']].tolist()
                                            )       
                                                )           
    ### To get the linear regression parameters use the line below
    # LCoef = reg.coef_

    # real-time measurment of of proxy customers
    datetimes = customer.data.index
    proxy_meas_repo = [ hist_data_proxy_customers[i].data[input_features['Forecasted_param']][
            (datetime.strptime(input_features['Last-observed-window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])).strftime('%Y-%m-%d %H:%M:%S'):
            (datetime.strptime(input_features['Last-observed-window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted'])).strftime('%Y-%m-%d %H:%M:%S')] for i in hist_data_proxy_customers.keys()]

    proxy_meas_repo_ = np.transpose(np.array(proxy_meas_repo))

    pred =  pd.DataFrame(reg.predict(proxy_meas_repo_),columns=['pred'])
    pred['datetime']= proxy_meas_repo[0].index
    
    nmi = [customer.nmi] * len(pred)
    pred['nmi'] = nmi
    
    # pred.reset_index(inplace=True)
    pred.set_index(['nmi', 'datetime'], inplace=True)    

    return (pred)


# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter, one for each time-step in a day
# # ================================================================

def pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,time_hms,input_features):
    """
    pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,input_features)

    This function generates forecasting values for a customer at using the some customers real-time measurements as proxies for a target customer.
    It generates a linear function mapping the each time-step of real-time measurements from know customers to the same time-step target customer values.

    It requires four inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    The third input is the time-step we wish to foreacast.
    And, the fourth input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """

    training_set_repo = []
    for i in hist_data_proxy_customers.keys():
        df = copy(hist_data_proxy_customers[i].data[input_features['Forecasted_param']][input_features['Start training']:input_features['End training']])
        training_set_repo.append(df[(df.index.hour == time_hms.hour) & (df.index.minute == time_hms.minute) & (df.index.second == time_hms.second)])

    df = copy(customer.data[input_features['Forecasted_param']][input_features['Start training']:input_features['End training']])
    training_set_target = df[(df.index.hour == time_hms.hour) & (df.index.minute == time_hms.minute) & (df.index.second == time_hms.second)]

    reg = LinearRegression().fit(
                                    np.transpose(np.array(training_set_repo)),
                                    np.array(training_set_target)       
                                    )

    # # # ## To get the linear regression parameters use the line below
    # # LCoef = reg.coef_
    
    datetimes = customer.data.index
    proxy_set_repo = []
    for i in hist_data_proxy_customers.keys():
        df = copy(hist_data_proxy_customers[i].data[input_features['Forecasted_param']][
            (datetime.strptime(input_features['Last-observed-window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])).strftime('%Y-%m-%d %H:%M:%S'):
            (datetime.strptime(input_features['Last-observed-window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted'])).strftime('%Y-%m-%d %H:%M:%S')]) 
        proxy_set_repo.append(df[(df.index.hour == time_hms.hour) & (df.index.minute == time_hms.minute) & (df.index.second == time_hms.second)])

    proxy_set_repo_ = np.transpose(np.array(proxy_set_repo))


    pred =  pd.DataFrame(reg.predict(proxy_set_repo_),columns=['pred'])
    pred['datetime']= proxy_set_repo[0].index

    nmi = [customer.nmi] * len(pred)
    pred['nmi'] = nmi

    pred.set_index(['nmi', 'datetime'], inplace=True)   

    return pred

def forecast_lin_reg_proxy_measures_separate_time_steps(hist_data_proxy_customers,customer,input_features):
    """
    pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,input_features)

    This function generates forecasting values for a customer at using the some customers real-time measurements as proxies for a target customer.
    It combines the forecasting values generated by the function pred_each_time_step_repo_linear_reg_single_node() for each time-step of the day.

    It requires three inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    And, the third input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """
    
    # generates a pandas datetime index with the same resolution of the original data. The start and end values used in tt are arbitrarily.
    tt = pd.date_range(start='2022-01-01',end='2022-01-02',freq=input_features['data_freq'])[0:-1]
    pred = pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,tt[0],input_features)

    for t in range(1,len(tt)):
        pred_temp = pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,tt[t],input_features)
        pred = pd.concat([pred,pred_temp])

    pred.sort_index(level = 1,inplace=True)

    return (pred)


# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter
# # ================================================================

def forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers,customer,input_features):
    """
    forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers,customer,input_features)

    This function generates forecasting values for a customer at using the some customers real-time measurements as proxies for a target customer.
    It uses the same the sk-forecast built in function that allows to use exogenous variables when forecasting a target customer. 
    More about this function can be found in "https://joaquinamatrodrigo.github.io/skforecast/0.4.3/notebooks/autoregresive-forecaster-exogenous.html".

    It requires three inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    And, the third input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """    

    cus_rep = pd.concat([hist_data_proxy_customers[i].data.active_power for i in hist_data_proxy_customers.keys()],axis=1)
    for name in range(0,len(cus_rep.columns.values)):
        cus_rep.columns.values[name] = f'{cus_rep.columns.values[name]}_{name}'

    customer.forecaster = ForecasterAutoreg(
            regressor = make_pipeline(StandardScaler(), Ridge()),  
            lags      = input_features['Window size']     
        )

    customer.forecaster.fit(y    = customer.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']],
                            exog = cus_rep.loc[input_features['Start training']:input_features['End training']])

    datetimes = customer.data.index

    check_time_zone_ = has_timezone(customer.data[input_features['Forecasted_param']].index[0])
    if check_time_zone_ == True:
        new_index =  pd.date_range(
                        start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                        end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                        freq=input_features['data_freq'],
                        tz="Australia/Sydney").delete(-1)
    else:
        new_index =  pd.date_range(
                        start=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                        end=datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ timedelta(days=input_features['Windows to be forecasted']),
                        freq=input_features['data_freq']).delete(-1)

    customer.predictions_exog_rep = customer.forecaster.predict(steps = len(new_index),
                                                       last_window = customer.data[input_features['Forecasted_param']].loc[(datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']],
                                                       exog = cus_rep.loc[new_index[0]:new_index[-1]] ).to_frame().set_index(new_index)
    

    result = customer.predictions_exog_rep
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result



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
    model.pv = Var(model.Time,customers_with_pv, bounds=(0,1))
    model.absLoad = Var(model.Time, within=NonNegativeReals)
    model.demand = Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_p = Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_n = Var(model.Time,customers_with_pv,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t,i] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]]
    model.cons = Constraint(model.Time,customers_with_pv,rule=load_balance)

    def abs_Load_1(model,t,i):
        return model.absLoad[t] >= sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]] for i in customers_without_pv )/len(customers_without_pv)
    model.cons_abs1 = Constraint(model.Time,customers_with_pv,rule=abs_Load_1)

    def abs_Load_2(model,t,i):
        return model.absLoad[t] >=  sum(data_one_time.loc[i].load_active[datetimes[t]] for i in customers_without_pv )/len(customers_without_pv) - sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv)
    model.cons_abs2 = Constraint(model.Time,customers_with_pv,rule=abs_Load_2)

    # # Objective
    def obj_rule(model):
        return (  model.absLoad[t] + sum(model.pv[t,i]**2 for i in customers_with_pv)/len(customers_with_pv) )
    # def obj_rule(model):
    #     return (  sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]]/len(customers_without_pv) for i in customers_without_pv) 
    #             + sum(model.pv[t,i]**2 for i in customers_with_pv) 
    #             )
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
    model.pv = Var(model.Time,customers_with_pv, bounds=(0,1))
    model.absLoad = Var(model.Time, within=NonNegativeReals)
    model.demand = Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_p = Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_n = Var(model.Time,customers_with_pv,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t,i] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]]
    model.cons = Constraint(model.Time,customers_with_pv,rule=load_balance)

    def abs_Load_1(model,t,i):
        return model.absLoad[t] >= sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]] for i in customers_without_pv )/len(customers_without_pv)
    model.cons_abs1 = Constraint(model.Time,customers_with_pv,rule=abs_Load_1)

    def abs_Load_2(model,t,i):
        return model.absLoad[t] >=  sum(data_one_time.loc[i].load_active[datetimes[t]] for i in customers_without_pv )/len(customers_without_pv) - sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv)
    model.cons_abs2 = Constraint(model.Time,customers_with_pv,rule=abs_Load_2)

    # # Objective
    def obj_rule(model):
        return (  model.absLoad[t] + sum(model.pv[t,i]**2 for i in customers_with_pv)/len(customers_with_pv) )
    # def obj_rule(model):
    #     return (  sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]]/len(customers_without_pv) for i in customers_without_pv) 
    #             + sum(model.pv[t,i]**2 for i in customers_with_pv) 
    #             )
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

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)

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

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)



def pool_executor_parallel_knownPVS(function_name,repeat_iter,input_features,customers_known_pv,datetimes):
    
    global customers_known_pv_shared

    customers_known_pv_shared = copy(customers_known_pv)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,repeat(datetimes)))  
    return results


def SDD_known_pvs_multiple_nodes(customers,input_features,customers_known_pv,datetimes):


    global customers_known_pv_shared

    predictions_prallel = pool_executor_parallel_knownPVS(SDD_known_pvs_single_node_for_parallel,customers.values(),input_features,customers_known_pv,datetimes)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    if 'customers_known_pv_shared' in globals():
        del(customers_known_pv_shared)

    return(predictions_prallel)


# # ================================================================
# # Technique 6: Weather Data
# # ================================================================
def SDD_using_temp_single_node(customer,data_weather):
    
    weather_input = data_weather[['AirTemp','hour','minute','Temp_EWMA','isweekend']]
    
    pv_dis = copy(customer.data.active_power)
    pv_dis[pv_dis > 0 ] = 0 
    pv_dis = -pv_dis
    load_dis = customer.data.active_power + pv_dis

    iteration = 0
    pv_dis_iter = copy(pv_dis*0)
    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 15:

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

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)

def SDD_using_temp_single_node_for_parallel(customer):

    print(f'customer_ID: {customer.nmi}')

    weather_input = shared_weather_data[['AirTemp','hour','minute','Temp_EWMA','isweekend']]

    pv_dis = copy(customer.data.active_power)
    pv_dis[pv_dis > 0 ] = 0 
    pv_dis = -pv_dis
    load_dis = customer.data.active_power + pv_dis

    iteration = 0
    pv_dis_iter = copy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 15:

        iteration += 1
        pv_dis_iter = copy(pv_dis)

        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=customer.data.index)
        load_dis[load_dis < 0 ] = 0 
        pv_dis = load_dis - customer.data.active_power

    pv_dis[pv_dis < 0 ] = 0 
    load_dis =  customer.data.active_power + pv_dis

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)

def pool_executor_parallel_temperature(function_name,repeat_iter,input_features,data_weather):
    
    global shared_weather_data

    shared_weather_data = copy(data_weather)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter))  
    return results

def SDD_using_temp_multilple_nodes(customers,input_features,data_weather):

    global shared_weather_data

    predictions_prallel = pool_executor_parallel_temperature(SDD_using_temp_single_node_for_parallel,customers.values(),input_features,data_weather)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    if 'shared_weather_data' in globals():
        del(shared_weather_data)

    return(predictions_prallel)


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
    
    weather_input = data_weather[['AirTemp','hour','minute','Temp_EWMA','isweekend']]
    
    pv_iter0 = copy(customer.data.active_power)
    pv_iter0[pv_iter0 > 0 ] = 0 
    pv_iter0 = -pv_iter0

    pv_dis = SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,pv_iter0)
    load_dis = customer.data.active_power + pv_dis
    
    iteration = 0
    pv_dis_iter = copy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 15:

        iteration += 1
        pv_dis_iter = copy(pv_dis)
        print(f'Iteration: {iteration}')
        
        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=pv_dis.index)
        pv_dis = load_dis - customer.data.active_power
        pv_dis[pv_dis < 0 ] = 0 
        pv_dis = SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,pv_dis)
        load_dis = customer.data.active_power + pv_dis

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)


def SDD_known_pvs_temp_single_node_algorithm_for_parallel(customer,datetimes):
    
    weather_input = shared_weather_data[['AirTemp','hour','minute','Temp_EWMA','isweekend']]

    pv_iter0 = copy(customer.data.active_power)
    pv_iter0[pv_iter0 > 0 ] = 0 
    pv_iter0 = -pv_iter0

    pv_dis = SDD_known_pvs_temp_single_node(customer,shared_data_known_pv,datetimes,pv_iter0)
    
    print(f'customer_ID: {customer.nmi} begin')
    load_dis = customer.data.active_power + pv_dis


    iteration = 0
    pv_dis_iter = copy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 15:
        
        iteration += 1
        pv_dis_iter = copy(pv_dis)
        # print(iteration)
        
        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=pv_dis.index)
        pv_dis = SDD_known_pvs_temp_single_node(customer,shared_data_known_pv,datetimes,load_dis - customer.data.active_power)
        load_dis = customer.data.active_power + pv_dis

    print(f'customer_ID: {customer.nmi} done!')

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)


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
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    if 'shared_data_known_pv' in globals():
        del(shared_data_known_pv)
    if 'shared_weather_data' in globals():
        del(shared_weather_data)

    return(predictions_prallel)
