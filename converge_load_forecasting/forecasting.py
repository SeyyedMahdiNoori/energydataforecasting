import pandas as pd
import numpy as np
import copy
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  HistGradientBoostingRegressor, RandomForestRegressor
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
import skforecast 
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
import datetime
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import multiprocess as mp
import tqdm
from functools import partialmethod
import itertools
# import connectorx as cx
import tspiral
import dateutil
from dateutil.parser import parse
from dateutil.parser import ParserError
from typing import Union, Dict, Tuple, List, Any
import statsmodels.api as sm
import pytz
from statsmodels.tools.sm_exceptions import InfeasibleTestError
import xgboost
import math
import mapie
from threadpoolctl import threadpool_limits
from prophet import Prophet
import joblib
import os




# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# A function to decide whether a string in the form of datetime has a time zone or not
def has_timezone(string: Union[str,pd.Timestamp]) -> bool:
    '''
    has_timezone(string) accept string in the form of datetime and return True if it has timezone, and it returns False otherwise.
    '''
    
    try:
        if type(string) == str:
            parsed_date = parse(string)
            return parsed_date.tzinfo is not None
        elif type(string) == pd._libs.tslibs.timestamps.Timestamp:
            return string.tzinfo is not None
        else:
            return False
    except (TypeError, ValueError):
        return False

def encoding_cyclical_time_features(time_series_data_index: pd.DatetimeIndex) -> pd.DataFrame:
    '''
    encoding_cyclical_time_features(time_series_data_index: pd.DatetimeIndex) -> pd.DataFrame
    This function accepts a pd.DatetimeIndex varibale and returns a pd.DataFrame with two columns. The output can be used to encode cyclical time features trigonometric functions.
    '''

    if type(time_series_data_index) is not pd.core.indexes.datetimes.DatetimeIndex:
        raise TypeError("time_series_data_index must be a pandas DatetimeIndex") 

    exog_time = pd.DataFrame({'datetime': time_series_data_index})
    exog_time = exog_time.set_index(time_series_data_index)
    
    # Add cyclical time featurs to the dataframe
    exog_time['minute_sin'] = np.sin(2 * np.pi * (exog_time.datetime.dt.hour*60 + exog_time.datetime.dt.minute) / 1440)
    exog_time['minute_cos'] = np.cos(2 * np.pi * (exog_time.datetime.dt.hour*60 + exog_time.datetime.dt.minute) / 1440)
    # exog_time['weekday'] = (time_series_data_index.day_of_week > 4).astype(int)     # ** Temporary ** Experiments have shown that if the training set is too small, weekday information has a negative impact on the accuracy of the forecast.
    exog_time.drop('datetime', axis=1, inplace=True)

    return exog_time

def data_frequency_in_minutes(time_series_data_index: pd.DatetimeIndex) -> int:
    
    if type(time_series_data_index) is not pd.core.indexes.datetimes.DatetimeIndex:
        raise TypeError("time_series_data_index must be a pandas DatetimeIndex") 
    
    try:
        return pd.Timedelta(time_series_data_index.freqstr).total_seconds()/60
    except ValueError:
        return pd.Timedelta('1' + time_series_data_index.freqstr).total_seconds()/60 

def prepare_proxy_data_for_training(time_series_data_index: pd.DatetimeIndex, data_proxy: pd.DataFrame) -> pd.DataFrame:

    if type(time_series_data_index) is not pd.core.indexes.datetimes.DatetimeIndex:
        raise TypeError("time_series_data_index must be a pandas DatetimeIndex")
    elif type(data_proxy) is not pd.DataFrame:
        raise TypeError("data_proxy must be a pandas DataFrame")
    
    # check the frequncy of the time_series and the data_proxy
    data_proxy_freq_int = data_frequency_in_minutes(data_proxy.index)
    time_series_data_freq_int = data_frequency_in_minutes(time_series_data_index)

    # Get the data_proxy to have the same frequncy as the time_series and fill the missing values.
    if data_proxy_freq_int ==  time_series_data_freq_int:
        exog_proxy = data_proxy.loc[time_series_data_index[0]:time_series_data_index[-1]]
    elif data_proxy_freq_int <  time_series_data_freq_int:
        exog_proxy = data_proxy.loc[time_series_data_index[0]:time_series_data_index[-1]].resample(time_series_data_index.freqstr).mean()
    else:
        exog_proxy = data_proxy.loc[time_series_data_index[0]:time_series_data_index[-1]].resample(time_series_data_index.freqstr).pad()

    return exog_proxy

def generate_output_adjusted_format_for_predictions(result: pd.DataFrame, customer, input_features: Dict) -> pd.DataFrame:
    '''
    generate_output_adjusted_format_for_predictions(result: pd.DataFrame, customer, input_features: Dict) -> pd.DataFrame
    This function takes on the a DataFrame from the prediction functions, and generate a multi-index (nmi,datetime) DataFrame. 
    '''
    
    # Adding bounds to the forecasting values 
    max_data = 1.5 * float(customer.data[input_features['Forecasted_param']].max())
    min_data = float(customer.data[input_features['Forecasted_param']].min())
    if min_data < 0:
        min_data = 1.5 * min_data
    else:
        min_data = 0.5 * min_data

    result.clip(lower=min_data,upper=max_data,inplace=True)

    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.rename_axis('datetime',inplace = True)
    result.set_index('nmi',append = True,inplace=True)
    result = result.swaplevel()

    return result

def select_regressor(regressor: str, loss_function: Union[sklearn.pipeline.Pipeline, None] = None) -> sklearn.pipeline.Pipeline:
    '''
    select_regressor(regressor: str, loss_function: Union[sklearn.pipeline.Pipeline, None] = None) -> sklearn.pipeline.Pipeline
    Select the regressor based on the user input
    '''

    if loss_function is None:
        loss_function = sklearn.linear_model.Ridge()

    if regressor == 'LinearRegression':
        regressor = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), loss_function)
    elif regressor == 'XGBoost':
        regressor = sklearn.pipeline.make_pipeline(xgboost.XGBRegressor()) # More specific objective functions: regressor = sklearn.pipeline.make_pipeline(xgboost.XGBRegressor(eval_metric = sklearn.metrics.mean_absolute_error))
    elif regressor == 'RandomForest':
        regressor = sklearn.pipeline.make_pipeline(RandomForestRegressor(random_state=123))

    return regressor

def select_loss_function(loss_function_input: str) -> sklearn.pipeline.Pipeline:
    '''
    select_loss_function(loss_function_input: str) -> sklearn.pipeline.Pipeline
    Select the loss function based on the user input
    '''
    from sklearn.linear_model import Ridge
    if loss_function_input == 'ridge':
        loss_function = sklearn.linear_model.Ridge()
    elif loss_function_input == 'lasso':
        loss_function = sklearn.linear_model.Lasso()
    elif loss_function_input == 'MSE':
        loss_function =  sklearn.linear_model.LinearRegression()

    return loss_function

def fill_in_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    '''
    fill_in_missing_data(data: pd.DataFrame) -> pd.DataFrame
    
    This function takes dataFrame with datetimeIndex. It fills the missing rows in the dateTime index
    and fills the added data values with zeros.
    '''

    try:
        if data.index.inferred_freq is None:
            if data.index[-10:].inferred_freq is None:
                freq = '5T'
            else:
                freq = data.index[-10:].inferred_freq
        else:
            freq = data.index.inferred_freq
    except Exception:
        freq = '5T'

    set_diff = pd.date_range(start=data.index[0],
                                   end=data.index[-1],
                                   freq=freq)
    df_new = data.reindex(set_diff)
    df_new = df_new.interpolate(method = 'linear')

    return df_new

def save_forecaster_to_file(data, filepath):
    # Save the list to disk using joblib.dump
    joblib.dump(data, filepath)

def load_forecaster_from_file(filepath):
    # Load the list from disk using joblib.load
    loaded_forecaster = joblib.load(filepath)
    return loaded_forecaster

def fill_input_dates_per_customer(data: pd.DataFrame, input_features: Dict) -> Tuple[str, str, str, str, pd.DataFrame, str, int]:
    '''
    fill_input_dates_per_customer(customer_data: pd.DataFrame, input_features: Dict) -> Tuple[str, str, str, str, pd.DataFrame, str, int]
    
    This function is used within the Customers class __init__ function. Considering the user preferences and the input data for each customer
    it generates start_training, end_training, last_observed_window, window_size, data, data_freq, steps_to_be_forecasted. These values are then
    will be assigned to each customer.
    '''

    # Data forequency.
    if data.index.freq == None:
        data = fill_in_missing_data(data)
    else:
        data.index.freq = data.index.inferred_freq

    # The datetime index that training starts from
    if input_features['start_training'] is None:
        start_training = data.index[0].strftime("%Y-%m-%d %H:%M:%S")
    else:
        start_training = max(input_features['start_training'],data.index[0].strftime("%Y-%m-%d %H:%M:%S"))

    # The last datetime index used for trainning.
    if input_features['end_training'] is None:
        end_training = data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    else:
        end_training = min(input_features['end_training'],data.index[-1].strftime("%Y-%m-%d %H:%M:%S"))

    # The last obersved window. The forecasting values are generated after this time index.
    if input_features['last_observed_window'] is None:
        last_observed_window = end_training
    else:
        last_observed_window = min(end_training,data.index[-1].strftime("%Y-%m-%d %H:%M:%S"))

    # Size of each window to be forecasted. A window is considered to be a day, and the resolution of the data is considered as the window size.
    # For example, for a data with resolution 30th minutely, the window size woul be 48.
    if input_features['window_size'] is None:
        window_size = min( int(datetime.timedelta(days = 1) / (data.index[1] - data.index[0])), int(len(data)/2) )
    else:
        window_size = min(input_features['window_size'], int(len(data)/2) )

    # days to be forecasted
    if input_features['days_to_be_forecasted'] is None:
        
        if len(input_features['date_to_be_forecasted']) == 10:
            delta = datetime.datetime.strptime(input_features['date_to_be_forecasted'],'%Y-%m-%d') - datetime.datetime.strptime(last_observed_window,'%Y-%m-%d %H:%M:%S')
        else:
            delta = datetime.datetime.strptime(input_features['date_to_be_forecasted'],'%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(last_observed_window,'%Y-%m-%d %H:%M:%S')

        try:
            freq_str = pd.to_timedelta(data.index.freqstr)
        except Exception:
            freq_str = pd.to_timedelta('1' + data.index.freqstr)

        steps_to_be_forecasted = math.floor(delta.total_seconds() / freq_str.total_seconds())
    
    else:

        try:
            freq_str = pd.to_timedelta(data.index.freqstr)
        except Exception:
            freq_str = pd.to_timedelta('1' + data.index.freqstr)

        steps_to_be_forecasted = math.floor( ( input_features['days_to_be_forecasted'] * 24 * 3600)  / freq_str.total_seconds())

    if steps_to_be_forecasted < 0:
        steps_to_be_forecasted = math.floor( ( 24 * 3600)  / pd.to_timedelta(data.index.freqstr).total_seconds())
        
    # data frequency
    data_freq = data.index.inferred_freq

    if start_training >= end_training:
        raise ValueError('Start training and end training do not match the input data. Left them blank in case not sure about the source of the problem.')
    if last_observed_window < end_training:
        raise ValueError('End training or last observed window do not match the input data. Left them blank in case not sure about the source of the problem.')

    return start_training, end_training, last_observed_window, window_size, data, data_freq, steps_to_be_forecasted

def proxy_input_data_cleaner(raw_proxy_data, input_features, tzinfo):

    try:        
        raw_proxy_data.rename(columns={"PeriodStart": "datetime"},inplace=True)
    except:
        pass
    
    try:
        raw_proxy_data = raw_proxy_data.drop('PeriodEnd', axis=1)
    except:
        pass

    try:
        raw_proxy_data = raw_proxy_data.drop('Period', axis=1)
    except:
        pass
    

    # # ###### Pre-process the data ######
    # format datetime to pandas datetime format
    try:
        check_time_zone = has_timezone(raw_proxy_data.datetime[0])
    except AttributeError:
        raise ValueError('Input data is not the correct format. It should have a column with "datetime", a column with name "nmi" and at least one more column which is going to be forecasted')

    try:
        
        if check_time_zone == False:
            raw_proxy_data['datetime'] = pd.to_datetime(raw_proxy_data['datetime'])
        else:
            raw_proxy_data['datetime'] = pd.to_datetime(raw_proxy_data['datetime'], utc=True, infer_datetime_format=True).dt.tz_convert(input_features['time_zone'])

    except ParserError:
        raise ValueError('data.datetime should be a string that can be meaningfully changed to time.')

    raw_proxy_data.set_index('datetime', inplace=True)

    if tzinfo is None:
        raw_proxy_data = raw_proxy_data.tz_localize(None)
    else:
        if check_time_zone == True:
            raw_proxy_data = raw_proxy_data.tz_convert(tzinfo)
        else:
            try:
                raw_proxy_data = raw_proxy_data.tz_localize(tzinfo)
            except Exception:
                raw_proxy_data = raw_proxy_data.tz_localize(tzinfo, ambiguous='infer')
    
    raw_proxy_data = raw_proxy_data.dropna(axis = 1)
    
    raw_proxy_data = raw_proxy_data[~raw_proxy_data.index.duplicated(keep='first')]

    raw_proxy_data = fill_in_missing_data(raw_proxy_data)

    return raw_proxy_data


# # ================================================================
# # Generate a class where its instances are the customers' nmi
# # ================================================================

# Customers is a class. An instant is assigned to each customer with its data. Most of the forecasting functions are then defined as methods within this class.
class Customers:

    instances = []

    def __init__(self, nmi, data, input_features):

        self.nmi = nmi      # store nmi in each object              

        # initialse each customer based on the user preferences and the input data
        [self.start_training, self.end_training,
          self.last_observed_window, self.window_size,
            self.data, self.data_freq, self.steps_to_be_forecasted] = fill_input_dates_per_customer(data.loc[self.nmi],input_features)    

        Customers.instances.append(self.nmi)

        Customers.check_time_zone_class = has_timezone(data.index.levels[1][0])

        Customers.regressor = input_features['regressor']

    @classmethod
    def remove_all_instances(cls):
        '''
        remove_all_instances(cls)
        
        This function removes all instances from the Customers class.
        It will be called at initialised function to prevent carrying instances when using the initialise function several times
        during a run.
        '''

        while cls.instances:
            instance = cls.instances.pop()
            del instance

    def generator_forecaster_object(self, input_features: Dict) -> None:
        '''
        generator_forecaster_object(input_features)

        This function takes a dictionary generated by the initialised function, named input_features, as input and assign a forecating object
        to the self customer instance.
        '''
        with threadpool_limits(limits=1, user_api='blas'):

            if input_features['algorithm'] == 'iterated' and input_features['probabilistic_algorithm'] == 'bootstrap':

                self.forecaster = ForecasterAutoreg(
                        regressor = input_features['regressor'],  
                        lags      = self.window_size     
                    )
                
                self.forecaster.fit(y = self.data.loc[self.start_training:self.end_training][input_features['Forecasted_param']],
                                    exog = self.exog)
            
            elif input_features['algorithm'] == 'iterated':

                self.forecaster = tspiral.forecasting.ForecastingCascade(
                estimator = input_features['regressor'],
                lags = range(1,self.window_size+1),
                use_exog = input_features['time_proxy'],
                accept_nan = False
                                    )
                
                self.forecaster.fit(self.exog, self.data.loc[self.start_training:self.end_training][input_features['Forecasted_param']])            

            elif input_features['algorithm'] == 'direct':

                self.forecaster = tspiral.forecasting.ForecastingChain(
                estimator = input_features['regressor'],
                n_estimators =  self.window_size,
                lags = range(1,self.window_size+1),
                use_exog = input_features['time_proxy'],
                accept_nan = False
                                    )
                
                self.forecaster.fit(self.exog, self.data.loc[self.start_training:self.end_training][input_features['Forecasted_param']])

            elif input_features['algorithm'] == 'rectified':
                
                if len(self.data.loc[self.start_training:self.end_training][input_features['Forecasted_param']]) - self.steps_to_be_forecasted - self.window_size <= 1:
                    test_size = None
                else:
                    test_size = self.steps_to_be_forecasted

                self.forecaster = tspiral.forecasting.ForecastingRectified(
                            estimators = input_features['regressor'],
                            n_estimators = self.window_size,
                            test_size = test_size,
                            lags = range(1, self.window_size + 1),
                            use_exog = input_features['time_proxy']
                                                )
                self.forecaster.fit(self.exog, self.data.loc[self.start_training:self.end_training][input_features['Forecasted_param']])

            elif input_features['algorithm'] == 'stacking':

                if len(self.data.loc[self.start_training:self.end_training][input_features['Forecasted_param']]) - self.steps_to_be_forecasted - self.window_size <= 1:
                    test_size = None
                else:
                    test_size = self.steps_to_be_forecasted

                self.forecaster = tspiral.forecasting.ForecastingStacked(
                                    input_features['regressor'],
                                    test_size = test_size,
                                    lags = range(1, self.window_size + 1),
                                    use_exog = input_features['time_proxy']
                                                        )
                self.forecaster.fit(self.exog, self.data.loc[self.start_training:self.end_training][input_features['Forecasted_param']])

            elif input_features['algorithm'] == 'prophet':

                self.forecaster = Prophet()

                if self.data.index.tzinfo is not None:
                    data_in = pd.DataFrame({'ds': self.data.index.tz_convert(None), 'y':self.data[input_features['Forecasted_param']].values})
                else:
                    data_in = pd.DataFrame({'ds': self.data.index, 'y':self.data[input_features['Forecasted_param']].values})
                
                if self.exog is not None:
                    exog = self.exog.reset_index()
                    exog.drop('index', axis = 1,inplace = True)
                    data_in = pd.concat([data_in,exog],axis=1)

                self.forecaster.fit(data_in)
            
            if input_features['save_forecaster_path'] is not None:
                save_forecaster_to_file(self.forecaster, input_features['save_forecaster_path'] + '/' + self.nmi + '.joblib')

    def generate_prediction(self, input_features: Dict) -> None:
        """
        generate_prediction(self, input_features: Dict) -> None
        
        This function takes a dictionary generated by the initialised function, named input_features, as input and assign the prediction values
        to the self customer instance.
        """

        # generate datetime index for the predicted values based on the window size and the last obeserved window.
        # new_index = generate_index_for_predicted_values(self.check_time_zone_class, input_features, self.data.index[1] - self.data.index[0])

        if input_features['algorithm'] == 'iterated' and input_features['probabilistic_algorithm'] == 'bootstrap':

            self.predictions = self.forecaster.predict(steps = len(self.new_index),
                                                        # last_window = self.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(self.last_observed_window,"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=self.days_to_be_forecasted)).strftime("%Y-%m-%d %H:%M:%S"):self.last_observed_window],
                                                        exog = self.exog_f).to_frame().set_index(self.new_index)

        elif input_features['algorithm'] == 'prophet':

            # future = self.forecaster.make_future_dataframe(periods = len(self.new_index), freq = self.data.index.freqstr).iloc[-len(self.new_index):]
            future = pd.DataFrame({'ds': self.exog_f.index})
            
            if self.exog_f is not None:
                exog_f = self.exog_f.reset_index()
                exog_f.drop('index', axis = 1,inplace = True)
                exog_f.index = future.index
                future = pd.concat([future,exog_f],axis=1)

            self.predictions = self.forecaster.predict(future)
            self.predictions = pd.DataFrame({'datetime': self.predictions['ds'] , input_features['Forecasted_param']: self.predictions['yhat']})

            if self.data.index.tzinfo is not None:
                self.predictions['datetime'] = pd.to_datetime( self.predictions['datetime'], utc=True, infer_datetime_format=True).dt.tz_convert(input_features['time_zone'])

            self.predictions = self.predictions.set_index('datetime')

        else:

            self.predictions = pd.DataFrame(self.forecaster.predict(self.f_steps
                                                                    ),index = self.new_index ,columns=['pred'])
        
    def generate_interval_prediction(self,input_features):
        """
        generate_interval_prediction(self,input_features)
        
        This function outputs three sets of values (a lower bound, an upper bound and the most likely value), using a recursive multi-step probabilistic forecasting method.
        The confidence level can be set in the function parameters as "interval = [10, 90]".
        """  
        with threadpool_limits(limits=1, user_api='blas'):
            # [10 90] considers 80% (90-10) confidence interval ------- n_boot: Number of bootstrapping iterations used to estimate prediction intervals.
            if input_features['algorithm'] == 'iterated' and input_features['probabilistic_algorithm'] == 'bootstrap':
                self.interval_predictions = self.forecaster.predict_interval(steps=len(self.new_index),
                                                                            interval = [10, 90],
                                                                            n_boot = 400,
                                                                            # last_window = self.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(self.last_observed_window,"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['days_to_be_forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):self.last_observed_window],
                                                                            exog = self.exog_f
                                                                            ).set_index(self.new_index)


            elif input_features['algorithm'] == 'prophet':

                future = self.forecaster.make_future_dataframe(periods = len(self.new_index), freq = self.data.index.freqstr).iloc[-len(self.new_index):]
                
                if self.exog_f is not None:
                    exog_f = self.exog_f.reset_index()
                    exog_f.drop('index', axis = 1,inplace = True)
                    exog_f.index = future.index
                    future = pd.concat([future,exog_f],axis=1)

                self.interval_predictions = self.forecaster.predict(future)
                self.interval_predictions = pd.DataFrame({'datetime': self.interval_predictions['ds'] , 'pred': self.interval_predictions['yhat'],'upper_bound': self.interval_predictions['yhat_upper'],'lower_bound': self.interval_predictions['yhat_lower'] })

                if self.data.index.tzinfo is not None:
                    self.interval_predictions['datetime'] = pd.to_datetime( self.interval_predictions['datetime'], utc=True, infer_datetime_format=True).dt.tz_convert(input_features['time_zone'])

                self.interval_predictions = self.interval_predictions.set_index('datetime')


            else:
            
                model = mapie.regression.MapieRegressor(self.forecaster, cv="prefit")
                model.fit(self.exog,self.data[input_features['Forecasted_param']].loc[self.exog.index])
                model.single_estimator_ = self.forecaster
                pred, bounds = model.predict(self.exog_f, alpha=0.2)
                bounds = bounds.reshape(len(pred),2)

                self.interval_predictions = pd.DataFrame({'pred': pred, 'lower_bound': bounds[:,0], 'upper_bound': bounds[:,1]}, index=self.exog_f.index)

    def generate_optimised_forecaster_object(self,input_features):            
        """
        generate_optimised_forecaster_object(self,input_features)
        
        This function generates a forecaster object for each \textit{nmi} to be used for a recursive multi-step forecasting method.
        It builds on function Generate\_forecaster\_object by combining grid search strategy with backtesting to identify the combination of lags 
        and hyperparameters that achieve the best prediction performance. As default, it is based on a linear least squares with \textit{l2} regularisation method. 
        Alternatively, it can use LinearRegression() and Lasso() methods to generate the forecaster object.
        """
        
        with threadpool_limits(limits=1, user_api='blas'):

            # This line is used to hide the bar in the optimisation process
            tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)

            self.forecaster = ForecasterAutoreg(
                    regressor = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.Ridge()),
                    lags      = input_features['window_size']      # The used data set has a 30-minute resolution. So, 48 denotes one full day window
                )

            # Regressor's hyperparameters
            param_grid = {'ridge__alpha': np.logspace(-3, 5, 10)}
            # Lags used as predictors
            lags_grid = [list(range(1,24)), list(range(1,48)), list(range(1,72))]

            # optimise the forecaster
            skforecast.model_selection.grid_search_forecaster(
                            forecaster  = self.forecaster,
                            y           = self.data.loc[self.start_training:self.end_training][input_features['Forecasted_param']],
                            param_grid  = param_grid,
                            # lags_grid   = lags_grid,
                            steps       =  self.window_size,
                            metric      = 'mean_absolute_error',
                            # refit       = False,
                            initial_train_size = len(self.data.loc[self.start_training:self.end_training][input_features['Forecasted_param']]) - self.steps_to_be_forecasted,
                            # fixed_train_size   = False,
                            return_best = True,
                            verbose     = False
                    )

    def generate_disaggregation_using_reactive(self,input_features):
        '''
        generate_disaggregation_using_reactive()

        Dissaggregate the solar generation and demand from the net real power measurement at the connection point.
        This approach uses reactive power as an indiction. More about this approach can be found in "Customer-Level Solar-Demand Disaggregation: The Value of Information".
        '''
        
        QP_coeff = (self.data.reactive_power.between_time('0:00','5:00')/self.data.active_power.between_time('0:00','5:00')[self.data.active_power.between_time('0:00','5:00') > 0.001]).resample('D').mean()
        QP_coeff[pd.Timestamp((QP_coeff.index[-1] + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))] = QP_coeff[-1]
        QP_coeff = QP_coeff.resample(self.data.index.freq).ffill()
        QP_coeff = QP_coeff.drop(QP_coeff.index[-1])
        QP_coeff = QP_coeff[QP_coeff.index <= self.data.reactive_power.index[-1]]

        set_diff = list( set(QP_coeff.index)-set(self.data.reactive_power.index) )
        QP_coeff = QP_coeff.drop(set_diff)

        load_est = self.data.reactive_power / QP_coeff 
        pv_est = load_est  - self.data.active_power
        pv_est[pv_est < 0] = 0
        load_est = pv_est + self.data.active_power
        
        return load_est, pv_est

    def Generate_disaggregation_positive_minimum_PV(self,input_features):
        '''
        generate_disaggregation_using_reactive()
        
        Dissaggregate the solar generation and demand from the net real power measurement at the connection point.
        This approach uses the negative and possitive values in the net measurement and assumes the minimum possible PV generation values.
        More about this approach can be found in "Customer-Level Solar-Demand Disaggregation: The Value of Information".
        '''

        return self.data[input_features["Forecasted_param"]].clip(lower = 0), - self.data[input_features["Forecasted_param"]].clip(upper = 0)


class Initialise_output:
            def __init__(self, customers, input_features, customers_nmi, datetimes, data_proxy) -> None:
                self.customers = customers
                self.input_features = input_features
                self.customers_nmi = customers_nmi
                self.datetimes = datetimes
                self.data_proxy = data_proxy

# class Initialise_output:
#             def __init__(self, data, customers, input_features, customers_nmi, datetimes) -> None:
#                 self.customers = customers
#                 self.data = data
#                 self.input_features = input_features
#                 self.customers_nmi = customers_nmi
#                 self.datetimes = datetimes

def read_data(customersdatapath: Union[str, None], raw_data: Union[pd.DataFrame, None]) -> pd.DataFrame:

    # Read data
    if customersdatapath is not None:
        data: pd.DataFrame = pd.read_csv(customersdatapath)     
    elif raw_data is not None:
        data = raw_data
    # elif db_url is not None and db_table_names is not None:
    #     sql = [f"SELECT * from {table}" for table in db_table_names]
    #     data = cx.read_sql(db_url,sql)
    #     data.sort_values(by='datetime',inplace=True)
    else:
        raise ValueError('Either customersdatapath, raw_data or db_url needs to be provided')
        
    return data

def input_features_time_zone(time_zone: Union[str, None]) -> str:
    if time_zone is None:
        time_zone = 'Australia/Sydney'
    elif time_zone not in pytz.all_timezones:
        raise ValueError(f"{time_zone} is NOT a valid time zone. only timezone that are in pytz.all_timezones are accpeted.")

    return time_zone

def format_datetime(check_time_zone: bool, data: pd.DataFrame, input_features: Dict) -> pd.DataFrame:
    try:
        if check_time_zone == False:
            data['datetime'] = pd.to_datetime(data['datetime'])
        else:
            data['datetime'] = pd.to_datetime(data['datetime'], utc=True, infer_datetime_format=True).dt.tz_convert(input_features['time_zone'])
    except Exception:
        raise ValueError('data.datetime should be a string that can be meaningfully changed to time.')
    
    return data

def read_proxy_data(proxydatapath: Union[str, None], raw_proxy_data: Union[pd.DataFrame, None], input_features: Dict, tzinfo: Union[str, None] ) -> Union[pd.DataFrame, None]:

    if proxydatapath is None and raw_proxy_data is None:
        data_proxy: Union[pd.DataFrame, None] = None
    elif proxydatapath is not None:
        raw_proxy_data = pd.read_csv(proxydatapath)
        data_proxy = proxy_input_data_cleaner(raw_proxy_data = raw_proxy_data, input_features = input_features, tzinfo = tzinfo)
    else:
        data_proxy = proxy_input_data_cleaner(raw_proxy_data = raw_proxy_data, input_features = input_features, tzinfo = tzinfo)
    
    return data_proxy

def input_features_forecast_param(forecasted_param: Union[None,str], columns: pd.core.indexes.base.Index) -> str:
    
    if forecasted_param is None:
        if 'active_power' in columns:
            forecasted_param = 'active_power'
        else:
            raise ValueError('forecasted_param needs to be provided or the input data should have a column name "active_power".')
    if forecasted_param not in columns:
        raise ValueError('forecasted_param is not in the data')
        
    return forecasted_param

def input_features_training_dates(training_dates: Union[str,None]) -> str:

    if training_dates is None:
        pass
    elif len(training_dates) == 10:
        try:
            datetime.datetime.strptime(training_dates,'%Y-%m-%d')
            training_dates = training_dates + ' ' + '00:00:00'
        except ValueError:
            raise ValueError(f"{training_dates} is NOT a valid date string.")
    elif len(training_dates) == 19:
        try:
            datetime.datetime.strptime(training_dates,'%Y-%m-%d %H:%M:%S')

        except ValueError:
            raise ValueError(f"{training_dates} is NOT a valid date string.")
    else:
        raise ValueError(f'{training_dates} does not have a correct format. It should be an string in "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S" or simply left blanck.')
    
    return training_dates

def input_features_window_size(window_size: Union[str,None]) -> str:

    if window_size is None or type(window_size) == int:
        pass
    else:
        raise ValueError('window size should be an integer')
    
    return window_size

def input_features_date_and_days_to_be_forecasted(days_to_be_forecasted: Union[None,str], date_to_be_forecasted: Union[None,str]) -> Tuple[str,str]:

    if days_to_be_forecasted is not None and date_to_be_forecasted is not None:
        raise ValueError('Only of the days_to_be_forecasted or date_to_be_forecasted should be given')

    elif days_to_be_forecasted is None and date_to_be_forecasted is None:     
        days_to_be_forecasted = 1

    elif type(days_to_be_forecasted) ==  int:
        pass
        
    elif type(date_to_be_forecasted) ==  str:
        
        days_to_be_forecasted = None

        date_to_be_forecasted = input_features_training_dates(date_to_be_forecasted)
    
    return days_to_be_forecasted, date_to_be_forecasted 

def input_features_core_usage(core_usage: Union[None,str]):
    
    if core_usage is None:
        core_usage = 8
    elif type(core_usage) == int:
        pass
    else:
        raise TypeError('Core usage should be an integer')

    return core_usage

    
def input_features_regressor_input(regressor: Union[None,str]) -> str:
    
    if regressor is None:
        regressor = 'LinearRegression'
    elif regressor == 'LinearRegression' or regressor == 'XGBoost' or regressor == 'RandomForest':
        pass
    else:
        raise ValueError(f"{regressor} is NOT a valid regressor. The regressor should be 'LinearRegression', 'XGBoost' or 'RandomForest'.")
    
    return regressor

def input_features_loss_function(loss_function: Union[None,str]) -> str:

    if loss_function is None:
        loss_function = 'ridge'
    elif loss_function == 'ridge' or loss_function == 'lasso' or loss_function == 'MSE':
        pass
    else:
        raise ValueError(f"{loss_function} is NOT a valid loss function. The loss_function should be 'ridge' or 'lasso' or 'MSE'.")
    
    return loss_function

def input_features_regressor(regressor: Union[None,str], loss_function: Union[None,str]) -> sklearn.pipeline.Pipeline:

    regressor = input_features_regressor_input(regressor=regressor)

    loss_function = input_features_loss_function(loss_function=loss_function)

    return select_regressor(regressor,select_loss_function(loss_function))


def input_features_time_proxy(time_proxy: Union[None,bool]) -> bool:
    
    if time_proxy is None or time_proxy == False:
        time_proxy = False
    elif time_proxy == True:
        pass
    else:
        raise TypeError(f"{time_proxy} is NOT valid. It should be True or False.")
    
    return time_proxy


def input_features_algorithm(algorithm: Union[None,str]) -> str:

    if algorithm is None:
        algorithm = 'iterated'
    elif algorithm == 'iterated' or algorithm == 'direct' or algorithm == 'rectified' or algorithm == 'stacking' or algorithm == 'prophet':
        pass
    else:
        raise ValueError(f"{algorithm} is NOT a valid algorithm. The algorithm should be 'iterated' or 'direct' or 'stacking' or 'rectified' or 'prophet'.")

    return algorithm


def input_features_run_sequentially(run_sequentially: Union[None,bool]) -> bool:

    if run_sequentially is None or run_sequentially == False:
        run_sequentially = False
    elif run_sequentially == True:
        pass
    else:
        raise TypeError(f"{run_sequentially} is NOT valid. It should be either True or False")
    
    return run_sequentially

def input_features_probabilistic_algorithm(probabilistic_algorithm: Union[None,str]) -> Tuple[str,None]:

    if probabilistic_algorithm is None or probabilistic_algorithm == 'bootstrap' or probabilistic_algorithm == 'jackknife':
        pass
    else:
        raise TypeError(f"{probabilistic_algorithm} is NOT valid. It should be 'bootstrap'  or 'jackknife' or left blank.")

    return probabilistic_algorithm

def input_features_save_forecaster_path(save_forecaster_path: Union[None,str], save_forecaster: Union[None,bool]) -> Tuple[str,None]:

    if save_forecaster_path is None:
        if save_forecaster is None or save_forecaster == False:
            save_forecaster_path = None
        elif save_forecaster == True:
            if os.path.exists('./forecaster_files') == False:
                os.makedirs('./forecaster_files')
            save_forecaster_path = './forecaster_files'
        else:
            raise ValueError(f"save_forecaster is {save_forecaster} instead of a Boolean input")
    else:
        if os.path.exists(save_forecaster_path) == True:
            pass 
        else:
            raise ValueError(f"{save_forecaster_path} does not exists save_forecaster is given something other than True")

    return save_forecaster_path

def check_data_nmi_datetime(data: pd.DataFrame) -> Tuple[bool,pd.DataFrame]:

    try:
        check_time_zone = has_timezone(data.datetime[0])
    except AttributeError:
        raise ValueError('Input data is not the correct format. It should have a column with "datetime", a column with name "nmi" and at least one more column which is going to be forecasted')

    if 'nmi' not in data.columns:
        data['nmi'] = ['1'] * len(data)

    return check_time_zone, data



# # ================================================================
# # Initialise the user preferences and pre-porcess the input data
# # ================================================================

def initialise(customersdatapath: Union[str, None] = None, raw_data: Union[pd.DataFrame, None] = None, forecasted_param: Union[str, None] = None,
                proxydatapath: Union[str, None] = None, raw_proxy_data: Union[pd.DataFrame, None] = None,
                start_training: Union[str, None] = None, end_training: Union[str, None] = None, last_observed_window: Union[str, None] = None,
                window_size: Union[int, None] = None, days_to_be_forecasted: Union[int, None] = None, date_to_be_forecasted: Union[int, None] = None,
                core_usage: Union[int, None] = None, db_url: Union[str, None] = None, db_table_names: Union[List[int], None] = None, regressor: Union[str, None] = None,
                loss_function: Union[str, None] = None, time_proxy: Union[bool, None] = None, algorithm: Union[str, None] = None,
                run_sequentially: Union[bool, None] = None, probabilistic_algorithm: Union[str, None] = None,
                time_zone: Union[str, None] = None, save_forecaster_path: Union[str, None] = None, save_forecaster: Union[bool, None] = None ) -> Union[Initialise_output,None]: 
    '''
    initialise(customersdatapath: Union[str, None] = None, raw_data: Union[pd.DataFrame, None] = None, forecasted_param: Union[str, None] = None,
                proxydatapath: Union[str, None] = None, raw_proxy_data: Union[pd.DataFrame, None] = None,
                start_training: Union[str, None] = None, end_training: Union[str, None] = None, nmi_type_path: Union[str, None] = None, last_observed_window: Union[str, None] = None,
                window_size: Union[int, None] = None, days_to_be_forecasted: Union[int, None] = None, core_usage: Union[int, None] = None,
                db_url: Union[str, None] = None, db_table_names: Union[List[int], None] = None) -> Union[Initialise_output,None]: 

    This function is to initialise the data and the input parameters required for the rest of the functions in this package. It requires one of the followings: 
    1. a path to a csv file 2. raw_data or 3. database url and the associate table names in that url. Other inputs are all optional.  
    '''

    try:
        Customers.remove_all_instances()
    except:
        pass

    try:
        # Read data
        data = read_data(customersdatapath=customersdatapath, raw_data=raw_data)
            
        # # ###### Pre-process the data ######
        check_time_zone, data = check_data_nmi_datetime(data=data)

        # create and populate input_features which is a paramter that will be used in almost all the functions in this package.
        # This paramtere represent the input preferenes. If there is no input to the initial() function to fill this parameters,
        # defeault values will be used to fill in the gap. 
        input_features: Dict[str, Union[str, bytes, bool, int, float, pd.Timestamp]] = {}

        input_features['time_zone'] = input_features_time_zone(time_zone=time_zone)
        
        # format datetime to pandas datetime format
        data = format_datetime(check_time_zone=check_time_zone, data=data, input_features=input_features)

        # Save customer nmis in a list
        customers_nmi = list(data['nmi'].unique())

        # Make datetime index of the dataset
        data.set_index(['nmi', 'datetime'], inplace=True)

        # save unique dates of the data
        datetimes: pd.DatetimeIndex  = pd.DatetimeIndex(data.index.unique('datetime')).sort_values()

        # read and process proxy data if it has been inputted
        data_proxy = read_proxy_data(proxydatapath=proxydatapath, raw_proxy_data=raw_proxy_data, input_features=input_features, tzinfo=data.index.levels[1].tzinfo)

        # The parameter to be forecasted. It should be a column name in the input data.
        input_features['Forecasted_param'] = input_features_forecast_param(forecasted_param=forecasted_param, columns=data.columns)

        # The datetime index that training starts from
        input_features['start_training'] = input_features_training_dates(training_dates=start_training)
        input_features['end_training'] = input_features_training_dates(training_dates=end_training)
        
        if last_observed_window is None:
            input_features['last_observed_window'] = input_features['end_training']
        else: 
            input_features['last_observed_window'] = input_features_training_dates(training_dates=last_observed_window)

        # Size of each window to be forecasted. A window is considered to be a day, and the resolution of the data is considered as the window size.
        # For example, for a data with resolution 30th minutely, the window size woul be 48.
        input_features['window_size'] = input_features_window_size(window_size=window_size)

        # The number of days to be forecasted.
        input_features['days_to_be_forecasted'], input_features['date_to_be_forecasted'] = input_features_date_and_days_to_be_forecasted(days_to_be_forecasted=days_to_be_forecasted, date_to_be_forecasted=date_to_be_forecasted)

        # number of processes parallel programming.
        input_features['core_usage'] = input_features_core_usage(core_usage=core_usage)

        # Check if the data is in the write formate for forecasting
        if data[input_features['Forecasted_param']].isna().any() == True:
            print('Warning!!! The data has Nan values')
        if not np.issubdtype(data[input_features['Forecasted_param']].dtype, np.number):
            print('Warning!!! The data does not have a integer or float type in the column which is going to be forecasted!')

        # Select loss regressor
        input_features['regressor'] = input_features_regressor(regressor=regressor, loss_function=loss_function)

        # Select whether to use time as proxy or not
        input_features['time_proxy'] = input_features_time_proxy(time_proxy=time_proxy)

        # Select a multi-step algorithm
        input_features['algorithm'] = input_features_algorithm(algorithm=algorithm)
        
        # Select whether to run the forecastin functions sequentially or parallelly
        input_features['run_sequentially'] = input_features_run_sequentially(run_sequentially=run_sequentially)
        
        # Select the probablictis 
        input_features['probabilistic_algorithm'] = input_features_probabilistic_algorithm(probabilistic_algorithm=probabilistic_algorithm)

        # Select if the forecaster is to be saved and its direcotry
        input_features['save_forecaster_path'] = input_features_save_forecaster_path(save_forecaster_path=save_forecaster_path,save_forecaster=save_forecaster)

        # A dictionary of all the customers with keys being customers_nmi and values being their associated Customers (which is a class) instance.
        customers = {customer: Customers(customer,data,input_features) for customer in customers_nmi}

        # data_initialised = Initialise_output(data, customers, input_features, customers_nmi, datetimes)
        data_initialised = Initialise_output(customers, input_features, customers_nmi, datetimes, data_proxy)
        
        return data_initialised

    except SystemExit:
        print('Exiting program')


# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Solar and Demand Forecasting functions
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================

def pool_executor_parallel(function_name, repeat_iter, input_features,data_proxy: Union[pd.DataFrame, None] = None):
    '''
    pool_executor_parallel(function_name,repeat_iter,input_features,data_proxy: Union[pd.DataFrame, None] = None)
    
    This function is used to parallelised the forecasting for each nmi
    '''
    
    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(input_features),itertools.repeat(data_proxy))) 
    return results


def generate_index_for_predicted_values(customer: Customers, input_features: Dict) -> pd.DatetimeIndex:
    '''
    generate_index_for_predicted_values(data_tzinfo: bool, input_features: Dict, deltatime: datetime.timedelta) -> pd.DatetimeIndex
    This function generate index values for the forecasting functions based on the values in the input_feature, partcularly how many windows should be forecasted and the last observed window.
    '''

    deltatime = customer.data.index[1]-customer.data.index[0]
    if customer.check_time_zone_class == True:
        return pd.date_range(
            start = datetime.datetime.strptime(customer.last_observed_window,"%Y-%m-%d %H:%M:%S") + deltatime,
            end = datetime.datetime.strptime(customer.last_observed_window,"%Y-%m-%d %H:%M:%S") + deltatime + deltatime * customer.steps_to_be_forecasted,
            freq = customer.data_freq,
            tz = input_features['time_zone']).delete(-1)
    else:
        return pd.date_range(
            start=datetime.datetime.strptime(customer.last_observed_window,"%Y-%m-%d %H:%M:%S") + deltatime,
            end=datetime.datetime.strptime(customer.last_observed_window,"%Y-%m-%d %H:%M:%S") + deltatime + deltatime * customer.steps_to_be_forecasted,
            freq=customer.data_freq).delete(-1)

# # ================================================================
# # Autoregressive recursive multi-step point-forecasting method
# # ================================================================

def add_exog_for_forecasting(customer: Customers, input_features: Dict, data_proxy: Union[pd.DataFrame, None] = None) -> None:
    '''
    add_exog_for_forecasting(customer: Customers, input_features: Dict) -> None
    This function generates cyclical time features for the class Customers to be used as an exogenous variable in the prediction algorithms or adds None in the class Customers
    if the user do not wish to consider these exogenous variables.
    '''
    
    customer.new_index = generate_index_for_predicted_values(customer, input_features)

    if input_features['time_proxy'] == True:
        customer.exog = encoding_cyclical_time_features(customer.data.loc[customer.start_training:customer.end_training].index)
        customer.exog_f = encoding_cyclical_time_features(customer.new_index)
        customer.f_steps = customer.exog_f
    else:
        customer.exog = None
        customer.exog_f = None
        customer.f_steps = np.arange(len(customer.new_index))

    if data_proxy is not None and customer.exog is not None:
        customer.exog = pd.concat([customer.exog,prepare_proxy_data_for_training(customer.data.loc[customer.start_training:customer.end_training].index,data_proxy)], axis = 1).interpolate(method = 'linear')
        customer.exog_f = pd.concat([customer.exog_f,prepare_proxy_data_for_training(customer.new_index,data_proxy)], axis = 1).interpolate(method = 'linear')
        customer.f_steps = customer.exog_f
    elif data_proxy is not None:
        customer.exog = prepare_proxy_data_for_training(customer.data.loc[customer.start_training:customer.end_training].index,data_proxy).interpolate(method = 'linear')
        customer.exog_f = prepare_proxy_data_for_training(customer.new_index,data_proxy).interpolate(method = 'linear')
        customer.f_steps = customer.exog_f

# This function outputs the forecasting for each nmi
def run_forecast_pointbased_single_node(customer: Customers, input_features: Dict, data_proxy: Union[pd.DataFrame, None] = None) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_single_node(customers_nmi,input_features)

    This function generates forecasting values for each customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is the customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    print("Customer nmi: {nmi}, {precent}%".format(nmi = customer.nmi, precent = round((Customers.instances.index(customer.nmi) + 1) / len(Customers.instances) * 100, 1)))
    # print("{pid}: Customer nmi: {first}, {ts}".format(pid=os.getpid(), first = customer.nmi, ts=datetime.datetime.now()))

    # Add exogonous variables (time and weekday information) to each customer if exog is selected True in the initialise function. Otherwise, it does nothin.
    add_exog_for_forecasting(customer, input_features, data_proxy)

    # Train a forecasting object
    customer.generator_forecaster_object(input_features)

    # Generate predictions 
    customer.generate_prediction(input_features)

    result = generate_output_adjusted_format_for_predictions(customer.predictions, customer, input_features)

    return result

def forecast_pointbased_single_node(customer: Customers, input_features: Dict, data_proxy: Union[pd.DataFrame, None] = None) -> pd.DataFrame:
    
    if input_features['probabilistic_algorithm'] is None:
        input_features['probabilistic_algorithm'] = 'bootstrap'
    
    return run_forecast_pointbased_single_node(customer, input_features, data_proxy)


def forecast_pointbased_multiple_nodes_parallel(customers: Dict[Union[int,str],Customers], input_features: Dict, data_proxy: Union[pd.DataFrame, None] = None) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    if input_features['probabilistic_algorithm'] is None:
        input_features['probabilistic_algorithm'] = 'bootstrap'

    predictions_prallel = pool_executor_parallel(run_forecast_pointbased_single_node,customers.values(),input_features, data_proxy)

    predictions_prallel = pd.concat(predictions_prallel, axis=0)
    predictions_prallel.index.levels[1].freq = predictions_prallel.index.levels[1].inferred_freq

    return predictions_prallel


def forecast_pointbased_multiple_nodes(customers: Dict[Union[int,str],Customers], input_features: Dict, data_proxy: Union[pd.DataFrame, None] = None) -> pd.DataFrame:

    if input_features['probabilistic_algorithm'] is None:
        input_features['probabilistic_algorithm'] = 'bootstrap'

    preds = [run_forecast_pointbased_single_node(customers[i],input_features,data_proxy) for i in customers.keys()]

    preds = pd.concat(preds, axis=0)
    preds.index.levels[1].freq = preds.index.levels[1].inferred_freq
    
    return preds

# # ================================================================
# # Recursive multi-step probabilistic forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_inetervalbased_single_node(customer: Customers, input_features: Dict, data_proxy: Union[pd.DataFrame, None] = None) -> pd.DataFrame:
    """
    forecast_inetervalbased_single_node(customers_nmi,input_features)

    This function generates forecasting values for each customer using the interval-based recursive multi-step forecasting method.
    It requires two inputs. The first input is the customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    print("Customer nmi: {nmi}, {precent}%".format(nmi = customer.nmi, precent = round((Customers.instances.index(customer.nmi) + 1) / len(Customers.instances) * 100, 1)))

    # Add exogonous variables (time and weekday information) to each customer if exog is selected True in the initialise function. Otherwise, it does nothin.
    input_features['time_proxy'] = True
    add_exog_for_forecasting(customer, input_features, data_proxy)
    
    # Train a forecasting object
    customer.generator_forecaster_object(input_features)
    
    # Generate interval predictions 
    customer.generate_interval_prediction(input_features)
    
    result = generate_output_adjusted_format_for_predictions(customer.interval_predictions, customer, input_features)

    return result

def forecast_inetervalbased_multiple_nodes_parallel(customers: Dict[Union[int,str],Customers], input_features: Dict, data_proxy: Union[pd.DataFrame, None] = None) -> pd.DataFrame:
    """
    forecast_inetervalbased_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the Interval-based recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    
    if input_features['probabilistic_algorithm'] is None:
        input_features['probabilistic_algorithm'] = 'jackknife'

    predictions_prallel = pool_executor_parallel(forecast_inetervalbased_single_node,customers.values(),input_features, data_proxy)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)
    # predictions_prallel.index.levels[1].freq = predictions_prallel.index.levels[1].inferred_freq

    return predictions_prallel

def forecast_inetervalbased_multiple_nodes(customers: Dict[Union[int,str],Customers], input_features: Dict, data_proxy: Union[pd.DataFrame, None] = None) -> pd.DataFrame:

    if input_features['probabilistic_algorithm'] is None:
        input_features['probabilistic_algorithm'] = 'jackknife'

    preds = [forecast_inetervalbased_single_node(customers[i], input_features, data_proxy) for i in customers.keys()]

    preds = pd.concat(preds, axis=0)
    preds.index.levels[1].freq = preds.index.levels[1].inferred_freq
    
    return preds

# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter as a exogenous variables
# # ================================================================

def check_corr_cause_proxy_customer(hist_data_proxy_customers : Dict[Union[int,str],Customers], customer: Customers, input_features: Dict, number_of_proxy_customers: int) -> pd.DataFrame:
    '''
    check_corr_cause_proxy_customer(hist_data_proxy_customers : Dict[Union[int,str],Customers], customer: Customers, input_features: Dict, number_of_proxy_customers: int) -> pd.DataFrame

    This function accepts a dictionary of participant customers, a non-participant customer object, a dictionary generated by the initialization function, and the maximum number
    of participant customers that will be used in the forecasting. The function starts by sorting the participant customers based on their correlation with the non-participant customer,
    on the values in the columns that need to be forecasted. Next, a Granger causality test is performed for each participant customer to determine their suitability for forecasting 
    the non-participant customer. Participants that fail the test are removed from the forecasting set. Finally, the function returns a DataFrame with the same Datetime index as the non-participant customer. 
    The DataFrame consists of multiple columns, each representing the participant customer data that has successfully passed the Granger causality test.
    '''
    
    if int(customer.data.index.freqstr[:-1]) == int(hist_data_proxy_customers[list(hist_data_proxy_customers.keys())[0]].data.index.freqstr[:-1]):
        cus_rep = pd.concat([pd.DataFrame(hist_data_proxy_customers[i].data['added']).rename(columns={'added': i}) for i in hist_data_proxy_customers.keys()],axis=1)
    elif int(customer.data.index.freqstr[:-1]) > int(hist_data_proxy_customers[list(hist_data_proxy_customers.keys())[0]].data.index.freqstr[:-1]):
        cus_rep = pd.concat([pd.DataFrame(hist_data_proxy_customers[i].data['added'].resample(customer.data.index.freqstr).mean()).rename(columns={'added': i}) for i in hist_data_proxy_customers.keys()],axis=1)
    else:
        cus_rep = pd.concat([pd.DataFrame(hist_data_proxy_customers[i].data['added'].resample(customer.data.index.freqstr).pad()).rename(columns={'added': i}) for i in hist_data_proxy_customers.keys()],axis=1)

    cus_rep = pd.concat([pd.DataFrame(customer.data.loc[customer.start_training:customer.end_training][input_features['Forecasted_param']]),cus_rep],axis=1)
    cus_rep = cus_rep.fillna(0)
    cus_rep.drop(input_features['Forecasted_param'],axis=1,inplace=True)

    corr = cus_rep.loc[customer.start_training:customer.end_training].corrwith(customer.data.loc[customer.start_training:customer.end_training][input_features['Forecasted_param']])
    corr = corr.fillna(0)
    corr.sort_values(ascending=False,inplace=True)

    corr = corr.head(min(number_of_proxy_customers,len(hist_data_proxy_customers.keys())))
    cus_rep = cus_rep[corr.to_dict()]

    for i in cus_rep.columns:
        try:
            if sm.tsa.stattools.grangercausalitytests(pd.concat([customer.data.loc[customer.start_training:customer.end_training][input_features['Forecasted_param']], cus_rep[i].loc[customer.start_training:customer.end_training]],axis=1
                                                                ), maxlag=1, verbose=False)[1][0]["ssr_ftest"][1] >= 0.05:
                cus_rep.drop(i,axis=1,inplace=True)   
                corr.drop(i,inplace=True) 

        except InfeasibleTestError:
            cus_rep.drop(i,axis=1,inplace=True)   
            corr.drop(i,inplace=True)
        # print(f'Warning: {customer.nmi} has values that are constant, or they follow an strange pattern that grangercausalitytests cannot be done on them!')
        # sm.tools.sm_exceptions

    corr = corr.head(min(number_of_proxy_customers,len(cus_rep.columns)))
    cus_rep = cus_rep[corr.to_dict()]

    return cus_rep

def forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers: Dict[Union[int,str],Customers], customer: Customers, input_features: Dict, number_of_proxy_customers: Union[int, None] = None, data_proxy: Union[pd.DataFrame, None] = None ) -> pd.DataFrame:
    """
    forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers: Dict[Union[int,str],Customers], customer: Customers, input_features: Dict, number_of_proxy_customers: Union[int, None] = None ) -> pd.DataFrame

    This function generates forecasting values for a customer at using the some customers real-time measurements as proxies for a target customer.
    It uses the same the sk-forecast built in function that allows to use exogenous variables when forecasting a target customer. 
    More about this function can be found in "https://joaquinamatrodrigo.github.io/skforecast/0.4.3/notebooks/autoregresive-forecaster-exogenous.html".

    It requires three inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    And, the third input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """    
    
    print("Customer nmi: {nmi}, {precent}%".format(nmi = customer.nmi, precent = round((Customers.instances.index(customer.nmi) + 1) / len(Customers.instances) * 100, 1)))

    if number_of_proxy_customers is None:
        number_of_proxy_customers = min(10,len(hist_data_proxy_customers.keys()))

    customers_proxy = check_corr_cause_proxy_customer(hist_data_proxy_customers, customer, input_features, number_of_proxy_customers)
    
    add_exog_for_forecasting(customer,input_features, data_proxy)

    if customer.exog is None and len(customers_proxy.columns) == 0:
        
        try:
            input_features['algorithm'] = 'rectified'

            if customer.exog is None and len(customers_proxy.columns) != 0:
                customer.exog = customers_proxy.loc[customer.start_training:customer.end_training]
                customer.exog_f = customers_proxy.loc[customer.new_index[0]:customer.new_index[-1]]
                customer.f_steps = customer.exog_f
            elif len(customers_proxy.columns) != 0 and customer.exog is not None:
                customer.exog = pd.concat([customer.exog,customers_proxy.loc[customer.exog.index]], axis = 1)
                customer.exog_f = pd.concat([customer.exog_f,customers_proxy.loc[customer.exog_f.index]], axis = 1)
                customer.f_steps = customer.exog_f

            customer.generator_forecaster_object(input_features)
            customer.generate_prediction(input_features)
            result = generate_output_adjusted_format_for_predictions(customer.predictions, customer, input_features)

        except Exception:
            input_features['algorithm'] = 'iterated'

            if customer.exog is None and len(customers_proxy.columns) != 0:
                customer.exog = customers_proxy.loc[customer.start_training:customer.end_training]
                customer.exog_f = customers_proxy.loc[customer.new_index[0]:customer.new_index[-1]]
                customer.f_steps = customer.exog_f
            elif len(customers_proxy.columns) != 0 and customer.exog is not None:
                customer.exog = pd.concat([customer.exog,customers_proxy.loc[customer.exog.index]], axis = 1)
                customer.exog_f = pd.concat([customer.exog_f,customers_proxy.loc[customer.exog_f.index]], axis = 1)
                customer.f_steps = customer.exog_f

            customer.generator_forecaster_object(input_features)
            customer.generate_prediction(input_features)
            result = generate_output_adjusted_format_for_predictions(customer.predictions, customer, input_features)            

    else:
        
        input_features['algorithm'] = 'iterated'
    
        if customer.exog is None and len(customers_proxy.columns) != 0:
            customer.exog = customers_proxy.loc[customer.start_training:customer.end_training]
            customer.exog_f = customers_proxy.loc[customer.new_index[0]:customer.new_index[-1]]
            customer.f_steps = customer.exog_f
        elif len(customers_proxy.columns) != 0 and customer.exog is not None:
            customer.exog = pd.concat([customer.exog,customers_proxy.loc[customer.exog.index]], axis = 1)
            customer.exog_f = pd.concat([customer.exog_f,customers_proxy.loc[customer.exog_f.index]], axis = 1)
            customer.f_steps = customer.exog_f

        customer.generator_forecaster_object(input_features)
        customer.generate_prediction(input_features)
        result = generate_output_adjusted_format_for_predictions(customer.predictions, customer, input_features)

    return result


def forecast_pointbased_exog_reposit_multiple_nodes(hist_data_proxy_customers: Dict[Union[int,str],Customers], n_customers: Dict[Union[int,str],Customers], input_features: Dict, number_of_proxy_customers: Union[int, None] = None, data_proxy: Union[pd.DataFrame, None] = None ) -> pd.DataFrame:

    if input_features['probabilistic_algorithm'] is None:
        input_features['probabilistic_algorithm'] = 'bootstrap'

    preds = [forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers,n_customers[i],input_features, number_of_proxy_customers,data_proxy) for i in n_customers.keys()]

    return pd.concat(preds, axis=0)



def pool_executor_parallel_exog(function_name, hist_data_proxy_customers, repeat_iter, input_features, number_of_proxy_customers,data_proxy):
    '''
    pool_executor_parallel(function_name,repeat_iter,input_features)
    
    This function is used to parallelised the forecasting for each nmi
    '''

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,itertools.repeat(hist_data_proxy_customers),repeat_iter,itertools.repeat(input_features),itertools.repeat(number_of_proxy_customers),itertools.repeat(data_proxy)))  
    return results

    # with ThreadPoolExecutor(max_workers=input_features['core_usage']) as executor:
    #     results = list(executor.map(function_name,itertools.repeat(hist_data_proxy_customers),repeat_iter,itertools.repeat(input_features),itertools.repeat(number_of_proxy_customers)))  
    # return results


def forecast_pointbased_exog_reposit_multiple_nodes_parallel(hist_data_proxy_customers: Dict[Union[int,str],Customers], customers: Dict[Union[int,str],Customers], input_features: Dict, number_of_proxy_customers: Union[int, None] = None, data_proxy: Union[pd.DataFrame, None] = None ) -> pd.DataFrame:
    """
    forecast_pointbased_exog_reposit_multiple_nodes_parallel(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    if input_features['probabilistic_algorithm'] is None:
        input_features['probabilistic_algorithm'] = 'bootstrap'

    predictions_prallel = pool_executor_parallel_exog(forecast_pointbased_exog_reposit_single_node,hist_data_proxy_customers,customers.values(),input_features,number_of_proxy_customers,data_proxy)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel

# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter as a exogenous variables
# # This function works on a data with two types, i.e., reposit and smart meters together
# # ================================================================

def forecast_mixed_type_customers(customers: Dict[Union[int,str],Customers], 
                                     participants: List[Union[int,str]], 
                                     input_features: Dict,
                                     end_participation_date: Union[datetime.datetime, pd.Timestamp, None] = None, 
                                     end_non_participants_date: Union[datetime.datetime, pd.Timestamp, None] = None,                                 
                                     non_participants:  Union[List[Union[int,str]], None] = None,
                                     number_of_proxy_customers: Union[int, None] = None,
                                     data_proxy: Union[pd.DataFrame, None] = None,
                                     proxy_complete: Union[bool, None] = None
                                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    '''
    forecast_mixed_type_customers(customers: Dict[Union[int,str],Customers], 
                                     participants: List[Union[int,str]], 
                                     input_features: Dict,
                                     end_participation_date: Union[datetime.datetime, pd.Timestamp, None] = None, 
                                     end_non_participants_date: Union[datetime.datetime, pd.Timestamp, None] = None,                                 
                                     non_participants:  Union[List[Union[int,str]], None] = None,
                                     number_of_proxy_customers: Union[int, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

    This function generates forecasting values for dataFrame consists of participant and non-participant customers in the trial.                                     
    '''

    
    # if the non-participants are not mentioned. All the customers that are not participating are considered as non-participant.
    if non_participants is None:
        non_participants = [i for i in customers.keys() if i not in participants]

    # create a dictionary of customers with the participant nmi as keys.
    customers_partipant = {}
    for i in participants:
        try:
            customers_partipant[i] = customers[i]
        except KeyError:
            try:
                customers_partipant[str(i)] = customers[str(i)]
            except KeyError:
                customers_partipant[int(i)] = customers[int(i)]
    
    # create a dictionary of customers with the non-participant nmi as keys.
    customers_non_participant = {i: customers[i] for i in non_participants}


    # update the inpute feature parameter, so that it matches the dates for the participant customers.
    if end_participation_date is not None:

        if type(end_participation_date) == str:
            # The datetime index that training starts from
            if len(end_participation_date) == 10:
                try:
                    datetime.datetime.strptime(end_participation_date,'%Y-%m-%d')
                    input_features['end_training'] = end_participation_date + ' ' + '00:00:00'
                except ValueError:
                    raise ValueError(f"{end_participation_date} is NOT a valid date string.")
            elif len(end_participation_date) == 19:
                try:
                    datetime.datetime.strptime(end_participation_date,'%Y-%m-%d %H:%M:%S')
                    input_features['end_training'] = end_participation_date
                except ValueError:
                    raise ValueError(f"{end_participation_date} is NOT a valid date string.")
            else:
                raise ValueError('end_participation_date does not have a correct format. It should be an string in "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S" or simply left blanck.')
        elif type(end_participation_date) == datetime.datetime:
            try:
                input_features['end_training'] = end_non_participants_date.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                input_features['end_training'] = end_non_participants_date.strftime("%Y-%m-%d")

        input_features['last_observed_window'] = input_features['end_training']

        for i in customers_partipant:
            [customers_partipant[i].start_training, customers_partipant[i].end_training,
            customers_partipant[i].last_observed_window, customers_partipant[i].window_size,
            customers_partipant[i].data, customers_partipant[i].data_freq, customers_partipant[i].steps_to_be_forecasted] = fill_input_dates_per_customer(customers_partipant[i].data,input_features)

    # generate forecate for participant customers.
    if proxy_complete == True:
    
        for i in customers_partipant.keys():
            customers_partipant[i].data = customers_partipant[i].data.rename(columns={input_features['Forecasted_param']: 'added'})
            participants_pred = None

    else:
        if input_features['run_sequentially'] == True:
            participants_pred = forecast_pointbased_multiple_nodes(customers_partipant, input_features, data_proxy)
        else:
            participants_pred = forecast_pointbased_multiple_nodes_parallel(customers_partipant, input_features, data_proxy)

        # combine forecast and historical data for participant customers.
        for i in participants_pred.index.levels[0]:

            temp = pd.DataFrame(pd.concat([pd.DataFrame(customers_partipant[i].data[input_features['Forecasted_param']]),participants_pred.loc[i]]))
            temp = temp[~temp.index.duplicated(keep='first')]
    
            customers_partipant[i].data = pd.concat([customers_partipant[i].data,
                                                    temp.rename(columns={input_features['Forecasted_param']: 'added'})], axis=1)


    # update the inpute feature parameter, so that it matches the dates for the non-participant customers.
    if end_non_participants_date is not None:

        if type(end_non_participants_date) == str:
            # The datetime index that training starts from
            if len(end_non_participants_date) == 10:
                try:
                    datetime.datetime.strptime(end_non_participants_date,'%Y-%m-%d')
                    input_features['end_training'] = end_non_participants_date + ' ' + '00:00:00'
                except ValueError:
                    raise ValueError(f"{end_non_participants_date} is NOT a valid date string.")
            elif len(end_non_participants_date) == 19:
                try:
                    datetime.datetime.strptime(end_non_participants_date,'%Y-%m-%d %H:%M:%S')
                    input_features['end_training'] = end_non_participants_date
                except ValueError:
                    raise ValueError(f"{end_non_participants_date} is NOT a valid date string.")
            else:
                raise ValueError('end_non_participants_date does not have a correct format. It should be an string in "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S" or simply left blanck.')
        elif type(end_non_participants_date) == datetime.datetime:
            try:
                input_features['end_training'] = end_non_participants_date.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                input_features['end_training'] = end_non_participants_date.strftime("%Y-%m-%d")

        input_features['last_observed_window'] = input_features['end_training']

        for i in customers_non_participant:
            [customers_non_participant[i].start_training, customers_non_participant[i].end_training,
            customers_non_participant[i].last_observed_window, customers_non_participant[i].window_size,
            customers_non_participant[i].data, customers_non_participant[i].data_freq, customers_non_participant[i].steps_to_be_forecasted] = fill_input_dates_per_customer(customers_non_participant[i].data,input_features)


    if number_of_proxy_customers is None:
        number_of_proxy_customers = min(10,len(participants))

    hist_data_proxy_customers = {i: customers_partipant[i] for i in list(customers_partipant.keys())[0:number_of_proxy_customers]}
    
    if input_features['run_sequentially'] == True:
        non_participants_pred = forecast_pointbased_exog_reposit_multiple_nodes(hist_data_proxy_customers, customers_non_participant, input_features, number_of_proxy_customers, data_proxy)
    else:
        non_participants_pred = forecast_pointbased_exog_reposit_multiple_nodes_parallel(hist_data_proxy_customers, customers_non_participant, input_features, number_of_proxy_customers, data_proxy)

    return participants_pred, non_participants_pred


# # ================================================================
# # Load_forecasting Using multi-series approach (multivariate and non-multivariate)
# # ================================================================

def forecast_multiseries_recusrive_autoregresive(n_customers,input_features):
    
    with threadpool_limits(limits=1, user_api='blas'):
        forecaster_ms = ForecasterAutoregMultiSeries(
                        regressor = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.Ridge()), # or use "regressor = HistGradientBoostingRegressor(random_state=123)," if the training set is big (n_sample >10000)
                        lags               = input_features['window_size'],
                        transformer_series = sklearn.preprocessing.StandardScaler(),
                    )
        
        forecaster_ms.fit(pd.DataFrame({i: n_customers[i].data.loc[input_features['start_training']:input_features['end_training']][input_features['Forecasted_param']] for i in n_customers.keys()}))

    time_zone_info = has_timezone(next(iter(n_customers.items()))[1].data.index[0])

    new_index = generate_index_for_predicted_values(time_zone_info, input_features, n_customers[list(n_customers.keys())[0]].data.index[1] - n_customers[list(n_customers.keys())[0]].data.index[0])
    pred_ = forecaster_ms.predict(steps=len(new_index),
                                # last_window = pd.DataFrame({i: n_customers[i].data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['last_observed_window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['days_to_be_forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['last_observed_window']] for i in n_customers.keys()})
                                ).set_index(new_index).rename_axis('datetime')

    return pd.melt(pred_.reset_index(),  id_vars='datetime' ,var_name='nmi', value_name='value').set_index(['nmi', 'datetime'])

def forecast_multivariate_recusrive_autoregresive_single_node(n_customers,customer,input_features,number_of_proxy_customers = None):

    # print(customer's nmi)
    print("Customer nmi: {nmi}, {precent}%".format(nmi = customer.nmi, precent = round((Customers.instances.index(customer.nmi) + 1) / len(Customers.instances) * 100, 1)))

    time_zone_info = has_timezone(customer.data.index[0])
    
    new_index = generate_index_for_predicted_values(time_zone_info, input_features, n_customers[list(n_customers.keys())[0]].data.index[1] - n_customers[list(n_customers.keys())[0]].data.index[0])
    
    with threadpool_limits(limits=1, user_api='blas'):
        forecaster = ForecasterAutoregMultiVariate(
                        regressor          = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.Ridge()),
                        level              = customer.nmi,
                        lags               = input_features['window_size'],
                        steps              = len(new_index)
                    )

        forecaster.fit(series=pd.DataFrame({i: n_customers[i].data.loc[input_features['start_training']:input_features['end_training']][input_features['Forecasted_param']] for i in n_customers.keys()}))
        
        result = forecaster.predict(steps=len(new_index),
                                    last_window = pd.DataFrame({i: n_customers[i].data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['last_observed_window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['days_to_be_forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['last_observed_window']] for i in n_customers.keys()})
                                    ).set_index(new_index).rename_axis('datetime')

    result.rename(columns={customer.nmi: input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result

def forecast_multivariate_recusrive_autoregresive_multiple_nodes(customers: Dict[Union[int,str],Customers], input_features: Dict, number_of_proxy_customers: Union[int, None] = None ) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """


    predictions_prallel = pool_executor_parallel_exog(forecast_multivariate_recusrive_autoregresive_single_node,customers,customers.values(),input_features,number_of_proxy_customers)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel

# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meters
# # ================================================================
def forecast_lin_reg_proxy_measures_single_node(hist_data_proxy_customers: Dict[Union[int,str],Customers], customer: Customers, input_features: Dict) -> pd.DataFrame:
    """
    forecast_lin_reg_proxy_measures(hist_data_proxy_customers,customer,input_features)

    This function generates forecasting values a customer using the some customers real-time measurements as proxies for a target customer.
    It generates a linear function mapping the real-time measurements from know customers to the target customer values.

    It requires three inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    And, the third input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """

    # Create a LinearRegression function from the historical data of proxy and target customers
    with threadpool_limits(limits=1, user_api='blas'):
        reg = sklearn.linear_model.LinearRegression().fit(
                                    np.transpose(np.array(
                                        [hist_data_proxy_customers[i].data[input_features['Forecasted_param']][input_features['start_training']:input_features['end_training']].tolist() for i in hist_data_proxy_customers.keys()]
                                                    )       ),
                                        np.array(customer.data[input_features['Forecasted_param']][input_features['start_training']:input_features['end_training']].tolist()
                                                )       
                                                    )           
        ### To get the linear regression parameters use the line below
    # LCoef = reg.coef_

    # real-time measurment of of proxy customers
    datetimes = customer.data.index
    proxy_meas_repo = [ hist_data_proxy_customers[i].data[input_features['Forecasted_param']][
            (datetime.datetime.strptime(input_features['last_observed_window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])).strftime('%Y-%m-%d %H:%M:%S'):
            (datetime.datetime.strptime(input_features['last_observed_window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])+ datetime.timedelta(days=input_features['days_to_be_forecasted'])).strftime('%Y-%m-%d %H:%M:%S')] for i in hist_data_proxy_customers.keys()]

    proxy_meas_repo_ = np.transpose(np.array(proxy_meas_repo))

    pred =  pd.DataFrame(reg.predict(proxy_meas_repo_),columns=[input_features['Forecasted_param']])
    pred['datetime']= proxy_meas_repo[0].index
    
    nmi = [customer.nmi] * len(pred)
    pred['nmi'] = nmi
    
    # pred.reset_index(inplace=True)
    pred.set_index(['nmi', 'datetime'], inplace=True)    

    return (pred)


# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter, one for each time-step in a day
# # ================================================================

def pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers: Dict[Union[int,str],Customers], customer: Customers, time_hms: pd.Timestamp, input_features: Dict) -> pd.DataFrame:
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
        df = copy.deepcopy(hist_data_proxy_customers[i].data[input_features['Forecasted_param']][input_features['start_training']:input_features['end_training']])
        training_set_repo.append(df[(df.index.hour == time_hms.hour) & (df.index.minute == time_hms.minute) & (df.index.second == time_hms.second)])

    df = copy.deepcopy(customer.data[input_features['Forecasted_param']][input_features['start_training']:input_features['end_training']])
    training_set_target = df[(df.index.hour == time_hms.hour) & (df.index.minute == time_hms.minute) & (df.index.second == time_hms.second)]

    with threadpool_limits(limits=1, user_api='blas'):
        reg = sklearn.linear_model.LinearRegression().fit(
                                        np.transpose(np.array(training_set_repo)),
                                        np.array(training_set_target)       
                                        )

    # # # ## To get the linear regression parameters use the line below
    # # LCoef = reg.coef_
    
    datetimes = customer.data.index
    proxy_set_repo = []
    for i in hist_data_proxy_customers.keys():
        df = copy.deepcopy(hist_data_proxy_customers[i].data[input_features['Forecasted_param']][
            (datetime.datetime.strptime(input_features['last_observed_window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])).strftime('%Y-%m-%d %H:%M:%S'):
            (datetime.datetime.strptime(input_features['last_observed_window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])+ datetime.timedelta(days=input_features['days_to_be_forecasted'])).strftime('%Y-%m-%d %H:%M:%S')]) 
        proxy_set_repo.append(df[(df.index.hour == time_hms.hour) & (df.index.minute == time_hms.minute) & (df.index.second == time_hms.second)])

    proxy_set_repo_ = np.transpose(np.array(proxy_set_repo))


    pred =  pd.DataFrame(reg.predict(proxy_set_repo_),columns=[input_features['Forecasted_param']])
    pred['datetime']= proxy_set_repo[0].index

    nmi = [customer.nmi] * len(pred)
    pred['nmi'] = nmi

    pred.set_index(['nmi', 'datetime'], inplace=True)   

    return pred

def forecast_lin_reg_proxy_measures_separate_time_steps(hist_data_proxy_customers: Dict[Union[int,str],Customers], customer: Customers, input_features: Dict) -> pd.DataFrame:
    """
    pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,input_features)

    This function generates forecasting values for a customer at using the some customers real-time measurements as proxies for a target customer.
    It combines the forecasting values generated by the function pred_each_time_step_repo_linear_reg_single_node() for each time-step of the day.

    It requires three inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    And, the third input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """
    
    # generates a pandas datetime index with the same resolution of the original data. The start and end values used in tt are arbitrarily.
    tt = pd.date_range(start='2022-01-01',end='2022-01-02',freq=customer.data.index.freqstr)[0:-1]
    pred = pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,tt[0],input_features)

    for t in range(1,len(tt)):
        pred_temp = pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,tt[t],input_features)
        pred = pd.concat([pred,pred_temp])

    pred.sort_index(level = 1,inplace=True)

    return (pred)


# # ================================================================
# # Long term load forecasting for RIT-D
# # ================================================================

def long_term_load_forecasting_single_node(customer: Customers, input_features: Dict, data_proxy: pd.DataFrame) -> pd.DataFrame:

    original_param = input_features["Forecasted_param"]

    # # Use iterated interval algorithm from skforecast
    input_features['algorithm'] = 'iterated' 
    input_features['probabilistic_algorithm'] == 'bootstrap'

    # # disaggregate solar and demand
    customer.data['demand'] = customer.data[input_features["Forecasted_param"]].clip(lower = 0)
    customer.data['solar'] = customer.data[input_features["Forecasted_param"]].clip(upper = 0)

    # # forecast demand
    print('demand prediction')
    input_features["Forecasted_param"] = 'demand'
    prediction_demand = forecast_inetervalbased_single_node(customer,input_features,data_proxy[['temp','humidity']])
    
     # # adjust solar predictions (making it zero if it is negative)
    prediction_demand.loc[customer.nmi][prediction_demand.loc[customer.nmi] < 0 ] = 0

    # # forecast solar
    print('solar prediction')
    input_features["Forecasted_param"] = 'solar'
    prediction_solar = forecast_inetervalbased_single_node(customer,input_features,data_proxy[['temp','humidity','solarradiation']])
    
    # # adjust solar predictions based on the solarradiation values (making it zero for early morning and late night time steps, and if it is positive)
    prediction_solar.loc[customer.nmi][prediction_solar.loc[customer.nmi] > 0 ] = 0
    prediction_solar.loc[customer.nmi][prepare_proxy_data_for_training(prediction_solar.loc[customer.nmi].index,data_proxy['solarradiation']) < 50 ] = 0

    # # adjust predictions based on the maximum and minmum values in the data
    demand_coeff = np.mean(customer.data['demand'].nlargest(10)) / np.mean(prediction_demand.demand.nlargest(10))
    solar_coeff = np.mean(customer.data['solar'].nsmallest(10))/ np.mean(prediction_solar.solar.nsmallest(10))
    
    if math.isnan(solar_coeff) == True:
        solar_coeff = 0

    if math.isnan(demand_coeff) == True:
        demand_coeff = 0

    # # Aggregate solar and demand
    input_features["Forecasted_param"] = original_param
    pred = prediction_demand.loc[customer.nmi].demand * demand_coeff + prediction_solar.loc[customer.nmi].solar * solar_coeff
    pred_lower = prediction_demand.loc[customer.nmi].lower_bound + prediction_solar.loc[customer.nmi].lower_bound * solar_coeff
    pred_upper = prediction_demand.loc[customer.nmi].upper_bound * demand_coeff + prediction_solar.loc[customer.nmi].upper_bound
    prediction = pd.DataFrame({input_features["Forecasted_param"]: pred.values, 'lower_bound': pred_lower.values, 'upper_bound': pred_upper.values}
                            , index= prediction_solar.index)

    return prediction



def long_term_load_forecasting_multiple_nodes_parallel(customers: Dict[Union[int,str],Customers], input_features: Dict, data_proxy: pd.DataFrame) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    predictions_prallel = pool_executor_parallel(long_term_load_forecasting_single_node,customers.values(),input_features, data_proxy)

    predictions_prallel = pd.concat(predictions_prallel, axis=0)
    predictions_prallel.index.levels[1].freq = predictions_prallel.index.levels[1].inferred_freq

    return predictions_prallel


def long_term_load_forecasting_multiple_nodes(customers: Dict[Union[int,str],Customers], input_features: Dict, data_proxy: Union[pd.DataFrame, None] = None) -> pd.DataFrame:


    preds = [long_term_load_forecasting_single_node(customers[i],input_features,data_proxy) for i in customers.keys()]

    preds = pd.concat(preds, axis=0)
    preds.index.levels[1].freq = preds.index.levels[1].inferred_freq
    
    return preds

def time_series_cross_validation(number_of_splits: int, customers: Dict[Union[int,str],Customers], input_features: Dict) -> pd.DataFrame:
    
    tscv = sklearn.model_selection.TimeSeriesSplit(n_splits=number_of_splits)
    generator = {nmi: tscv.split(customers[nmi].data) for nmi in customers.keys()}

    res = []
    for range_number in range(0,number_of_splits):

        for nmi in customers.keys():
            
            input_features["end_training"] = customers[nmi].data.index[next(generator[nmi])[0]][-1].strftime("%Y-%m-%d %H:%M:%S")
            
            [customers[nmi].start_training, customers[nmi].end_training,
                customers[nmi].last_observed_window, customers[nmi].window_size,
                    customers[nmi].data, customers[nmi].data_freq, customers[nmi].steps_to_be_forecasted] = fill_input_dates_per_customer(customers[nmi].data,input_features)
        
        if input_features['run_sequentially'] == True:
            prediction = forecast_pointbased_multiple_nodes(customers,input_features)
        else:
            prediction = forecast_pointbased_multiple_nodes_parallel(customers,input_features)

        res.append(prediction)

    res_com  = pd.concat(res,axis=1)
    new_column_names = [f'{res_com.columns[0]}_{i+1}' for i in range(len(res_com.columns))]
    res_com.columns = new_column_names

    return res_com

def near_cast(customers_nmis: list, forecasters_file_path: str, newly_measured_data: pd.DataFrame, days_to_be_forecasted: Union[str, None] = None,
            date_to_be_forecasted: Union[str, None] = None, forecasted_param: Union[str, None] = None, run_sequentially: bool = True, proxy_measure: Union[pd.DataFrame, None] = None) -> pd.DataFrame:

    files_in_dirctory = list(os.listdir(forecasters_file_path))
    forecaster_files = [x[:-7] for x in files_in_dirctory if '.joblib' in x]

    forecasters = {}
    for nmi in customers_nmis:
        if nmi in forecaster_files:
            forecasters[nmi] = load_forecaster_from_file(forecasters_file_path + '/'+ nmi + '.joblib')

    if run_sequentially == True:
        
        preds = [run_single_near_cast(forecasters, newly_measured_data, forecaster_files, nmi = i, days_to_be_forecasted = days_to_be_forecasted, forecasted_param = forecasted_param, date_to_be_forecasted = date_to_be_forecasted, proxy_measure = proxy_measure) for i in forecasters.keys()]
        preds = pd.concat(preds, axis=0)
        preds.index.levels[1].freq = preds.index.levels[1].inferred_freq
    
    else: 
    
        predictions_prallel = pool_executor_parallel_near_cast(run_single_near_cast, forecasters, newly_measured_data, forecaster_files, proxy_measure, forecasted_param, days_to_be_forecasted, date_to_be_forecasted)
        preds = pd.concat(predictions_prallel, axis=0)
        preds.index.levels[1].freq = preds.index.levels[1].inferred_freq

    return preds

def generate_index_near_cast(forecaster: Any, newly_measured_data: pd.DataFrame) -> pd.DatetimeIndex:

    return pd.date_range(   start = newly_measured_data.index[-forecaster.window_size],
                            end = newly_measured_data.index[-1],
                            freq = forecaster.last_window.index.inferred_freq)  

def generate_index_near_cast_exog(forecaster: Any, last_window_index: pd.DatetimeIndex, days_to_be_forecasted: Union[str, None] = None, date_to_be_forecasted: Union[str, None] = None) -> Tuple[pd.DatetimeIndex,int]:
    
    if days_to_be_forecasted is None:
                
        delta = pd.to_datetime(date_to_be_forecasted).tz_localize(last_window_index.tzinfo) - last_window_index[-1]
        
        try:
            freq_str = pd.to_timedelta(last_window_index.freqstr)
        except Exception:
            freq_str = pd.to_timedelta('1' + last_window_index.freqstr)

        steps_to_be_forecasted = math.floor(delta.total_seconds() / freq_str.total_seconds())
    
    else:

        try:
            freq_str = pd.to_timedelta(last_window_index.freqstr)
        except Exception:
            freq_str = pd.to_timedelta('1' + last_window_index.freqstr)

        steps_to_be_forecasted = math.floor( ( days_to_be_forecasted * 24 * 3600)  / freq_str.total_seconds())

    if steps_to_be_forecasted < 0:
        steps_to_be_forecasted = math.floor( ( 24 * 3600)  / pd.to_timedelta(last_window_index.freqstr).total_seconds())

    exog_index =  pd.date_range(    start = (last_window_index + forecaster.last_window.index.freq * forecaster.window_size )[0],
                                    end = (last_window_index+ forecaster.last_window.index.freq * steps_to_be_forecasted )[-1],
                                    freq = forecaster.last_window.index.inferred_freq)        
    
    return exog_index, steps_to_be_forecasted
     
def pool_executor_parallel_near_cast(function_name, forecasters: Any, newly_measured_data: pd.DataFrame, forecaster_files: List, proxy_measure: Union[pd.DataFrame, None] = None, forecasted_param: Union[str, None] = None, days_to_be_forecasted: Union[str, None] = None, date_to_be_forecasted: Union[str, None] = None):
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(function_name,itertools.repeat(forecasters),itertools.repeat(newly_measured_data),itertools.repeat(forecaster_files), list(forecasters.keys()),itertools.repeat(proxy_measure),itertools.repeat(forecasted_param),itertools.repeat(days_to_be_forecasted),itertools.repeat(date_to_be_forecasted))) 
    return results


def run_single_near_cast(forecasters: Any, newly_measured_data_nmi: pd.DataFrame, forecaster_files: List, nmi: str, proxy_measure: Union[pd.DataFrame, None] = None, forecasted_param: Union[str, None] = None, days_to_be_forecasted: Union[str, None] = None, date_to_be_forecasted: Union[str, None] = None) -> pd.DataFrame:
    
    with threadpool_limits(limits=1, user_api='blas'):

        print("Customer nmi: {nmi}, {precent}%".format(nmi = nmi, precent = round((list(forecasters.keys()).index(nmi) + 1) / len(list(forecasters.keys())) * 100, 1)))

        forecaster = forecasters[nmi]
        newly_measured_data = newly_measured_data_nmi.loc[nmi]

        last_window_index = generate_index_near_cast(forecaster, newly_measured_data)
        exog_index, steps_to_be_forecasted = generate_index_near_cast_exog(forecaster,last_window_index,days_to_be_forecasted,date_to_be_forecasted)

        exog_columns = forecaster.exog_col_names

        if exog_columns == None:
            exog = None
        elif 'minute_sin' in exog_columns:
            exog = encoding_cyclical_time_features(exog_index)
        
        try: 
            other_exog_columns = [x for x in exog_columns if x not in ['minute_sin', 'minute_cos']]
        except:
            other_exog_columns =  None

        if other_exog_columns is not None:
            
            for proxy_name in other_exog_columns:

                try:
    
                    if exog is None:
                        exog = proxy_measure.loc[proxy_name].reindex(exog_index, fill_value=0)
                    else:
                        exog = pd.concat([exog,(proxy_measure.loc[proxy_name].reindex(exog_index, fill_value=0)).rename(columns={'pred': proxy_name})], axis = 1)
                
                except:
                    raise ValueError(f'{proxy_name} is used in the training step but is not provided for the near cast function.')

        if forecasted_param is None:
            forecasted_param = 'active_power'
        
        if newly_measured_data.index.freq == None:
            newly_measured_data.index.freq = newly_measured_data.index.inferred_freq 
        
        result = pd.DataFrame(
                        forecaster.predict(
                            steps = steps_to_be_forecasted,
                            last_window = newly_measured_data.loc[last_window_index][forecasted_param],
                            exog = exog) )
        
        result['nmi'] = [nmi] * len(result)
        result.rename_axis('datetime',inplace = True)
        result.set_index('nmi',append = True,inplace=True)
        result = result.swaplevel()

        return result
            
def near_cast_mixed(participants: list, non_participants: list, forecasters_file_path: str, newly_measured_data: pd.DataFrame, days_to_be_forecasted: Union[str, None] = None,
                    date_to_be_forecasted: Union[str, None] = None, forecasted_param: Union[str, None] = None, run_sequentially: bool = True,
                    proxy_measure: Union[pd.DataFrame, None] = None) -> Tuple[pd.DataFrame,pd.DataFrame]:

    print("forecast participants ...")
    pred_participants =  near_cast(participants, forecasters_file_path, newly_measured_data, days_to_be_forecasted,
                             date_to_be_forecasted, forecasted_param,
                             run_sequentially, proxy_measure)
    
    print("forecast non-participants ...")
    pred_non_participants =  near_cast(non_participants, forecasters_file_path, newly_measured_data, days_to_be_forecasted,
                             date_to_be_forecasted, forecasted_param,
                             run_sequentially, proxy_measure = pred_participants)
    
    return pred_participants, pred_non_participants

def linear_increase(series: pd.Series, rate: float) -> pd.Series:
    result = pd.Series(index=series.index)
    initial_value = 0  # Initial value before linear increase
    current_value = initial_value
    consecutive_count = 0  # Counter for consecutive non-zero values

    for idx, val in enumerate(series):
        if val != 0:
            consecutive_count += 1
            result.iloc[idx] = current_value + (rate * (consecutive_count - 1))
        else:
            current_value = initial_value  # Reset to initial value
            result.iloc[idx] = current_value
            consecutive_count = 0  # Reset consecutive count

    return result

def post_prediction_heat_adjustment(pred:pd.DataFrame, exog:pd.DataFrame, high_temp_limit: float = 30, high_temp_initial: float = 25,
                                    max_heat_effect: float = 0.5, exponential_decay_factor: float =  0.01, solar_adjustment: float =  4) -> pd.DataFrame:
    
    temp = pd.DataFrame({"temperature": exog.temperature})
    temp_filt = pd.DataFrame({'temperature' : temp.temperature.apply(lambda x: 0 if x < high_temp_limit else 1)})
    temp_D = temp_filt.resample('D').max()
    temp_D = temp_D.fillna(0)
    temp_D['transformed'] = linear_increase(temp_D['temperature'],1)
    heat_temp = temp_D.transformed.resample('30T').ffill()
    heat = np.minimum( (1 - np.exp(np.minimum(-exponential_decay_factor*(exog.temperature - high_temp_initial),0))) * heat_temp, max_heat_effect)

    demand = pred.clip(lower = 0)
    solar = pred.clip(upper = 0)

    demand = heat.loc[pred.index] * demand + demand
    solar = heat.loc[pred.index]/solar_adjustment * solar + solar
    
    return demand + solar
