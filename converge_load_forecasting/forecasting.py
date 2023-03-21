import pandas as pd
import numpy as np
import copy
import sklearn
from sklearn.tree import  DecisionTreeRegressor
import skforecast 
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import datetime
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import multiprocess as mp
from pyomo.environ import NonNegativeReals, ConcreteModel, Var, Objective, Set, Constraint
from pyomo.opt import SolverFactory
import tqdm
from functools import partialmethod
import itertools
import connectorx as cx
import tsprial
import dateutil
from dateutil.parser import parse
from dateutil.parser import ParserError
from typing import Union, Dict, Tuple, List
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import InfeasibleTestError
import xgboost

# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# A function to decide whether a string in the form of datetime has a time zone or not
def has_timezone(string: str) -> bool:
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

# # ================================================================
# # Generate a class where its instances are the customers' nmi
# # ================================================================

# Customers is a class. An instant is assigned to each customer with its data. Most of the forecasting functions are then defined as methods within this class.
class Customers:

    num_of_customers = 0

    def __init__(self, nmi, data):

        self.nmi = nmi      # store nmi in each object              
        self.data = data.loc[self.nmi]      # store data in each object         

        Customers.check_time_zone_class = has_timezone(data.index.levels[1][0])

        Customers.num_of_customers += 1

    def generate_forecaster_autoregressive(self, input_features):            
        """
        generate_forecaster_autoregressive(input_features)
        
        This function generates a forecaster object to be used for a autoregressive recursive multi-step forecasting method. 
        It is based on a linear least squares with l2 regularization method. Alternatively, LinearRegression() and Lasso() that
        have different objective can be used with the same parameters.
        """

        # Create a forecasting object
        self.forecaster_autoregressive = ForecasterAutoreg(
                regressor = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.Ridge()),  
                lags      = input_features['Window size']      
            )
 
        # Train the forecaster using the train data
        self.forecaster_autoregressive.fit(y=self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']])

    def generate_forecaster_autoregressive_xgboost(self, input_features):            
        """
        generate_forecaster_autoregressive(input_features)
        
        This function generates a forecaster object to be used for a autoregressive recursive multi-step forecasting method. 
        It is based on a linear least squares with l2 regularization method. Alternatively, LinearRegression() and Lasso() that
        have different objective can be used with the same parameters.
        """

        # Create a forecasting object
        self.forecaster_autoregressive_xgboost = ForecasterAutoreg(
                regressor = xgboost.XGBRegressor(),  
                lags      = input_features['Window size']      
            )
 
        # Train the forecaster using the train data
        self.forecaster_autoregressive_xgboost.fit(y=self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']])

    def generate_forecaster_autoregressive_xgboost_time_exog(self, input_features):            
        """
        generate_forecaster_autoregressive(input_features)
        
        This function generates a forecaster object to be used for a autoregressive recursive multi-step forecasting method. 
        It is based on a linear least squares with l2 regularization method. Alternatively, LinearRegression() and Lasso() that
        have different objective can be used with the same parameters.
        """

        # Create a forecasting object
        self.forecaster_autoregressive_xgboost_time_exog = ForecasterAutoreg(
                regressor = xgboost.XGBRegressor(),  
                lags      = input_features['Window size']      
            )

        exog_time = pd.DataFrame({'datetime': self.data.loc[input_features['Start training']:input_features['End training']].index})
        exog_time = exog_time.set_index(self.data.loc[input_features['Start training']:input_features['End training']].index)
        exog_time['minute_sin'] = np.sin(2 * np.pi * (exog_time.datetime.dt.hour*60 + exog_time.datetime.dt.minute) / 1440)
        exog_time['minute_cos'] = np.cos(2 * np.pi * (exog_time.datetime.dt.hour*60 + exog_time.datetime.dt.minute) / 1440)
        exog_time.drop('datetime', axis=1, inplace=True)

        # Train the forecaster using the train data
        self.forecaster_autoregressive_xgboost_time_exog.fit(y = self.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']],
                                                   exog = exog_time.loc[input_features['Start training']:input_features['End training']])

    def generate_forecaster_direct(self,input_features):            
        """
        generate_forecaster_direct(self,input_features)
        
        This function generates a forecaster object to be used for a multi-step forecasting method. 
        More details about this approach can be found in "https://robjhyndman.com/papers/rectify.pdf" and "https://towardsdatascience.com/6-methods-for-multi-step-forecasting-823cbde4127a"
        """

        self.forecaster_direct = tsprial.forecasting.ForecastingChain(
                    sklearn.linear_model.Ridge(),
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

        self.forecaster_stacking = tsprial.forecasting.ForecastingStacked(
                    [sklearn.linear_model.Ridge(), DecisionTreeRegressor()],
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

        self.forecaster_rectified = tsprial.forecasting.ForecastingRectified(
                    sklearn.linear_model.Ridge(),
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
        tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)

        self.forecaster = ForecasterAutoreg(
                regressor = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.Ridge()),
                lags      = input_features['Window size']      # The used data set has a 30-minute resolution. So, 48 denotes one full day window
            )

        # Regressor's hyperparameters
        param_grid = {'ridge__alpha': np.logspace(-3, 5, 10)}
        # Lags used as predictors
        lags_grid = [list(range(1,24)), list(range(1,48)), list(range(1,72))]

        # optimise the forecaster
        skforecast.model_selection.grid_search_forecaster(
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
                                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                                        freq=input_features['data_freq'],
                                        tz="Australia/Sydney").delete(-1)
        else:
            new_index =  pd.date_range(
                            start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                            end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                            freq=input_features['data_freq']).delete(-1)

        self.predictions_autoregressive = self.forecaster_autoregressive.predict(steps=len(new_index), last_window=self.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']]).to_frame().set_index(new_index)
        

    def generate_prediction_autoregressive_xgboost(self,input_features):
        """
        generate_prediction_autoregressive_xgboost(self,input_features)
        
        This function outputs the prediction values using a Recursive autoregressive multi-step point-forecasting method.          
        """
        
        # generate datetime index for the predicted values based on the window size and the last obeserved window.
        if self.check_time_zone_class == True:
            new_index =  pd.date_range(
                                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                                        freq=input_features['data_freq'],
                                        tz="Australia/Sydney").delete(-1)
        else:
            new_index =  pd.date_range(
                            start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                            end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                            freq=input_features['data_freq']).delete(-1)

        self.predictions_autoregressive_xgboost = self.forecaster_autoregressive_xgboost.predict(steps=len(new_index), last_window=self.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']]).to_frame().set_index(new_index)

    def generate_prediction_autoregressive_xgboost_time_exog(self,input_features):
        """
        generate_prediction_autoregressive_xgboost(self,input_features)
        
        This function outputs the prediction values using a Recursive autoregressive multi-step point-forecasting method.          
        """
        
        # generate datetime index for the predicted values based on the window size and the last obeserved window.
        if self.check_time_zone_class == True:
            new_index =  pd.date_range(
                                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                                        freq=input_features['data_freq'],
                                        tz="Australia/Sydney").delete(-1)
        else:
            new_index =  pd.date_range(
                            start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                            end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                            freq=input_features['data_freq']).delete(-1)

        exog_time = pd.DataFrame({'datetime': new_index})
        # exog_time = exog_time.set_index(new_index)
        exog_time['minute_sin'] = np.sin(2 * np.pi * (exog_time.datetime.dt.hour*60 + exog_time.datetime.dt.minute) / 1440)
        exog_time['minute_cos'] = np.cos(2 * np.pi * (exog_time.datetime.dt.hour*60 + exog_time.datetime.dt.minute) / 1440)
        exog_time.drop('datetime', axis=1, inplace=True)

        self.predictions_autoregressive_xgboost_time_exog = self.forecaster_autoregressive_xgboost_time_exog.predict(steps=len(new_index),
                                                                                                                    last_window = self.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']],
                                                                                                                    exog = exog_time).to_frame().set_index(new_index)

    def generate_prediction_direct(self,input_features):
        """
        generate_prediction(self,input_features)
        
        This function outputs the prediction values using a Recursive direct multi-step point-forecasting method.  
        """
        
        # generate datetime index for the predicted values based on the window size and the last obeserved window.
        if self.check_time_zone_class == True:
            new_index =  pd.date_range(
                                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                                        freq=input_features['data_freq'],
                                        tz="Australia/Sydney").delete(-1)
        else:
            new_index =  pd.date_range(
                            start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                            end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
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
                                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                                        freq=input_features['data_freq'],
                                        tz="Australia/Sydney").delete(-1)
        else:
            new_index =  pd.date_range(
                            start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                            end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
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
                                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                                        freq=input_features['data_freq'],
                                        tz="Australia/Sydney").delete(-1)
        else:
            new_index =  pd.date_range(
                            start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                            end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
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
                                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                                        freq=input_features['data_freq'],
                                        tz="Australia/Sydney").delete(-1)
        else:
            new_index =  pd.date_range(
                            start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]),
                            end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (self.data.index[1]-self.data.index[0]) + datetime.timedelta(days=input_features['Windows to be forecasted']),
                            freq=input_features['data_freq']).delete(-1)       

        # [10 90] considers 80% (90-10) confidence interval ------- n_boot: Number of bootstrapping iterations used to estimate prediction intervals.
        self.interval_predictions = self.forecaster_autoregressive.predict_interval(steps=len(new_index),
                                                                        interval = [10, 90],
                                                                        n_boot = 1000,
                                                                        last_window = self.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']]
                                                                        ).set_index(new_index)
        # self.predictions                   = self.forecaster.predict(steps=len(new_index),                                    last_window=self.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']]).to_frame().set_index(new_index)


    def generate_disaggregation_using_reactive(self,input_features):
        '''
        generate_disaggregation_using_reactive()

        Dissaggregate the solar generation and demand from the net real power measurement at the connection point.
        This approach uses reactive power as an indiction. More about this approach can be found in "Customer-Level Solar-Demand Disaggregation: The Value of Information".
        '''
        
        QP_coeff = (self.data.reactive_power.between_time('0:00','5:00')/self.data.active_power.between_time('0:00','5:00')[self.data.active_power.between_time('0:00','5:00') > 0.001]).resample('D').mean()
        QP_coeff[pd.Timestamp((QP_coeff.index[-1] + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))] = QP_coeff[-1]
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

        D = copy.deepcopy(self.data.active_power)
        D[D<=0] = 0
        S = copy.deepcopy(self.data.active_power)
        S[S>=0] = 0
        self.data['pv_disagg'] =  - S
        self.data['demand_disagg'] = D


# # ================================================================
# # Initialise the user preferences and pre-porcess the input data
# # ================================================================

def initialise(customersdatapath: Union[str, None] = None, raw_data: Union[pd.DataFrame, None] = None, forecasted_param: Union[str, None] = None,
                weatherdatapath: Union[str, None] = None, raw_weather_data: Union[pd.DataFrame, None] = None,
                start_training: Union[str, None] = None, end_training: Union[str, None] = None, nmi_type_path: Union[str, None] = None, Last_observed_window: Union[str, None] = None,
                window_size: Union[int, None] = None, windows_to_be_forecasted: Union[int, None] = None, core_usage: Union[int, None] = None,
                db_url: Union[str, None] = None, db_table_names: Union[List[int], None] = None) -> Tuple[pd.DataFrame, Dict, Dict, List[Union[int,str]], List[pd.Timestamp]]:   #Tuple[pd.DataFrame,List[Union[int,str]],List[Union[int,str]],List[pd.Timestamp],Dict,Dict,Dict]
    '''
    initialise(customersdatapath=None,raw_data=None,forecasted_param=None,weatherdatapath=None,raw_weather_data=None,start_training=None,end_training=None,nmi_type_path=None,Last_observed_window=None,window_size=None,windows_to_be_forecasted=None,core_usage=None,db_url=None,db_table_names=None)

    This function is to initialise the data and the input parameters required for the rest of the functions in this package. It requires one of the followings: 
    1. a path to a csv file 2. raw_data or 3. database url and the associate table names in that url. Other inputs are all optional.  
    '''

    # Read data
    if customersdatapath is not None:
        data: pd.DataFrame = pd.read_csv(customersdatapath)     
    elif raw_data is not None:
        data = copy.deepcopy(raw_data)
    elif db_url is not None and db_table_names is not None:
        sql = [f"SELECT * from {table}" for table in db_table_names]
        data = cx.read_sql(db_url,sql)
        data.sort_values(by='datetime',inplace=True)
    else:
        print('Error!!! Either customersdatapath, raw_data or db_url needs to be provided')
        return pd.DataFrame(), {}, {}, [1], [pd.Timestamp('2017-01-01')] # To match the number of outputs (It was: pd.DataFrame(),[1],[1],[pd.Timestamp('2017-01-01')],{},{},{} )

    # # ###### Pre-process the data ######
    # format datetime to pandas datetime format
    try:
        check_time_zone = has_timezone(data.datetime[0])
    except AttributeError:
        print('Error!!! Input data is not the correct format! It should have a column with "datetime", a column with name "nmi" and at least one more column which is going to be forecasted')
        return pd.DataFrame(), {}, {}, [1], [pd.Timestamp('2017-01-01')] # To match the number of outputs

    try:
        if check_time_zone == False:
            data['datetime'] = pd.to_datetime(data['datetime'])
        else:
            data['datetime'] = pd.to_datetime(data['datetime'], utc=True, infer_datetime_format=True).dt.tz_convert("Australia/Sydney")
            # data["datetime"] = data["datetime"].dt.tz_convert("Australia/Sydney")
    except ParserError:
        print('Error!!! data.datetime should be a string that can be meaningfully changed to time.')
        return pd.DataFrame(), {}, {}, [1], [pd.Timestamp('2017-01-01')] # To match the number of outputs

    # # Add weekday column to the data
    # data['DayofWeek'] = data['datetime'].dt.day_name()

    # Save customer nmis in a list
    customers_nmi = list(dict.fromkeys(list(data['nmi'].values)))

    # Make datetime index of the dataset
    data.set_index(['nmi', 'datetime'], inplace=True)

    # save unique dates of the data
    datetimes: pd.DatetimeIndex  = pd.DatetimeIndex(data.index.unique('datetime'))

    # create and populate input_features which is a paramter that will be used in almost all the functions in this package.
    # This paramtere represent the input preferenes. If there is no input to the initial() function to fill this parameters,
    # defeault values will be used to fill in the gap. 
    input_features: Dict[str, Union[str, bytes, bool, int, float, pd.Timestamp]] = {}

    # The parameters to be forecasted. It should be a column name in the input data.
    if forecasted_param is None:
        input_features['Forecasted_param'] = 'active_power'
    else:
        input_features['Forecasted_param'] = forecasted_param

    # The datetime index that training starts from
    if start_training is None:
        input_features['Start training'] = datetimes[0].strftime("%Y-%m-%d %H:%M:%S")
    elif len(start_training) == 10:
        input_features['Start training'] = start_training + ' ' + '00:00:00'
    elif len(start_training) == 19:
        input_features['Start training'] = start_training
    else:
        print('Error!!! start training does not have a correct format. It should be an string in "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S" or simply left blanck.')
        return pd.DataFrame(), {}, {}, [1], [pd.Timestamp('2017-01-01')] # To match the number of outputs

    # The last datetime index used for trainning.
    if end_training is None:
        # input_features['End training'] = (datetimes[-1] - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        input_features['End training'] = (datetimes[-1]).strftime("%Y-%m-%d %H:%M:%S")
    elif len(end_training) == 10:
        input_features['End training'] = end_training + ' ' + '00:00:00'
    elif len(end_training) == 19:
        input_features['End training'] = end_training
    else:
        print('Error!!! end training does not have a correct format. It should be an string in "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S" or simply left blanck.')
        return pd.DataFrame(), {}, {}, [1], [pd.Timestamp('2017-01-01')] # To match the number of outputs

    # The last obersved window. The forecasting values are generated after this time index.
    if Last_observed_window is None:
        input_features['Last-observed-window'] = input_features['End training']
    elif len(Last_observed_window) == 10:
        input_features['Last-observed-window'] = Last_observed_window + ' ' + '00:00:00'
    elif len(Last_observed_window) == 19:
        input_features['Last-observed-window'] = Last_observed_window
    else:
        print('Error!!! last observed window does not have a correct format. It should be an string in "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S" or simply left blanck.')
        return pd.DataFrame(), {}, {}, [1], [pd.Timestamp('2017-01-01')] # To match the number of outputs


    # Select customers with pv. This is used to read the "nmi.csv" file, used in the Converge project.
    if nmi_type_path is None:
        customers_nmi_with_pv = copy.deepcopy(customers_nmi)
    else:
        input_features['nmi_type_path'] = nmi_type_path
        data_nmi = pd.read_csv(nmi_type_path)
        data_nmi.set_index(data_nmi['nmi'],inplace=True)

        customers_nmi_with_pv = [ data_nmi.loc[i]['nmi'] for i in customers_nmi if data_nmi.loc[i]['has_pv']==True ]
        data['has_pv']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['has_pv']] for i in customers_nmi]* len(datetimes)))
        data['customer_kind']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['customer_kind']] for i in customers_nmi]* len(datetimes)))
        data['pv_system_size']  = list(itertools.chain.from_iterable([ [data_nmi.loc[i]['pv_system_size']] for i in customers_nmi]* len(datetimes)))

    # Size of each window to be forecasted. A window is considered to be a day, and the resolution of the data is considered as the window size.
    # For example, for a data with resolution 30th minutely, the window size woul be 48.
    if window_size is None:
        input_features['Window size'] = int(datetime.timedelta(days = 1) / (datetimes[1] - datetimes[0]))
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

    try:
        if check_time_zone == True:
            datetimes.freq = input_features['data_freq']
            data.index.levels[1].freq = input_features['data_freq']
    except Exception:
        pass

    if data[input_features['Forecasted_param']].isna().any() == True or (data[input_features['Forecasted_param']].dtype != float and  data[input_features['Forecasted_param']].dtype != int):
        print('Warning!!! The data has Nan values or does not have a integer or float type in the column which is going to be forecasted!')

    # A dictionary of all the customers with keys being customers_nmi and values being their associated Customers (which is a class) instance.
    customers = {customer: Customers(customer,data) for customer in customers_nmi}

    return data, customers, input_features, customers_nmi, datetimes
    # return data, customers_nmi,customers_nmi_with_pv,datetimes, customers, input_features


# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Solar and Demand Forecasting functions
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================

def pool_executor_parallel(function_name, repeat_iter, input_features):
    '''
    pool_executor_parallel(function_name,repeat_iter,input_features)
    
    This function is used to parallelised the forecasting for each nmi
    '''
    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(input_features)))  
    return results


# # ================================================================
# # Autoregressive recursive multi-step point-forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_pointbased_autoregressive_single_node(customer: Customers, input_features: Dict) -> pd.DataFrame:
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
    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


def forecast_pointbased_autoregressive_multiple_nodes(customers: Dict[Union[int,str],Customers], input_features: Dict) -> pd.DataFrame:
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
# # Autoregressive recursive multi-step point-forecasting method using XGBoost as regressor
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_pointbased_autoregressive_xgboost_single_node(customer: Customers, input_features: Dict) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_single_node(customers_nmi,input_features)

    This function generates forecasting values for each customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is the customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    
    # print(customer's nmi)
    print(" Customer nmi: {first}".format(first = customer.nmi))

    # Train a forecasting object
    customer.generate_forecaster_autoregressive_xgboost(input_features)
    
    # Generate predictions 
    customer.generate_prediction_autoregressive_xgboost(input_features)

    result = customer.predictions_autoregressive_xgboost
    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result

def forecast_pointbased_autoregressive_xgboost_multiple_nodes(customers: Dict[Union[int,str],Customers], input_features: Dict) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    predictions_prallel = pool_executor_parallel(forecast_pointbased_autoregressive_xgboost_single_node,customers.values(),input_features)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel


# # ================================================================
# # Autoregressive recursive multi-step point-forecasting method using XGBoost as regressor and time as exogounos
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_pointbased_autoregressive_xgboost_time_exog_single_node(customer: Customers, input_features: Dict) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_single_node(customers_nmi,input_features)

    This function generates forecasting values for each customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is the customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    
    # print(customer's nmi)
    print(" Customer nmi: {first}".format(first = customer.nmi))

    # Train a forecasting object
    customer.generate_forecaster_autoregressive_xgboost_time_exog(input_features)
    
    # Generate predictions 
    customer.generate_prediction_autoregressive_xgboost_time_exog(input_features)

    result = customer.predictions_autoregressive_xgboost_time_exog
    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result

def forecast_pointbased_autoregressive_xgboost_multiple_nodes_time_exog(customers: Dict[Union[int,str],Customers], input_features: Dict) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """
    predictions_prallel = pool_executor_parallel(forecast_pointbased_autoregressive_xgboost_time_exog_single_node,customers.values(),input_features)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel


# # ================================================================
# # Recitifed recursive multi-step point-forecasting method
# # ================================================================

# This function outputs the forecasting for each nmi
def forecast_pointbased_rectified_single_node(customer: Customers, input_features: Dict) -> pd.DataFrame:
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
    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


def forecast_pointbased_rectified_multiple_nodes(customers: Dict[Union[int,str],Customers],input_features: Dict) -> pd.DataFrame:
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
def forecast_pointbased_direct_single_node(customer: Customers, input_features: Dict) -> pd.DataFrame:
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
    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


def forecast_pointbased_direct_multiple_nodes(customers: Dict[Union[int,str],Customers], input_features: Dict) -> pd.DataFrame:
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
def forecast_pointbased_stacking_single_node(customer: Customers, input_features: Dict) -> pd.DataFrame:
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
    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


def forecast_pointbased_stacking_multiple_nodes(customers: Dict[Union[int,str],Customers],input_features: Dict) -> pd.DataFrame:
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
def forecast_inetervalbased_single_node(customer: Customers, input_features: Dict) -> pd.DataFrame:
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
    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result

def forecast_inetervalbased_multiple_nodes(customers: Dict[Union[int,str],Customers], input_features: Dict) -> pd.DataFrame:
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
    reg = sklearn.linear_model.LinearRegression().fit(
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
            (datetime.datetime.strptime(input_features['Last-observed-window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])).strftime('%Y-%m-%d %H:%M:%S'):
            (datetime.datetime.strptime(input_features['Last-observed-window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])+ datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime('%Y-%m-%d %H:%M:%S')] for i in hist_data_proxy_customers.keys()]

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
        df = copy.deepcopy(hist_data_proxy_customers[i].data[input_features['Forecasted_param']][input_features['Start training']:input_features['End training']])
        training_set_repo.append(df[(df.index.hour == time_hms.hour) & (df.index.minute == time_hms.minute) & (df.index.second == time_hms.second)])

    df = copy.deepcopy(customer.data[input_features['Forecasted_param']][input_features['Start training']:input_features['End training']])
    training_set_target = df[(df.index.hour == time_hms.hour) & (df.index.minute == time_hms.minute) & (df.index.second == time_hms.second)]

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
            (datetime.datetime.strptime(input_features['Last-observed-window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])).strftime('%Y-%m-%d %H:%M:%S'):
            (datetime.datetime.strptime(input_features['Last-observed-window'],'%Y-%m-%d %H:%M:%S') + (datetimes[1]-datetimes[0])+ datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime('%Y-%m-%d %H:%M:%S')]) 
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
    tt = pd.date_range(start='2022-01-01',end='2022-01-02',freq=input_features['data_freq'])[0:-1]
    pred = pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,tt[0],input_features)

    for t in range(1,len(tt)):
        pred_temp = pred_each_time_step_repo_linear_reg_single_node(hist_data_proxy_customers,customer,tt[t],input_features)
        pred = pd.concat([pred,pred_temp])

    pred.sort_index(level = 1,inplace=True)

    return (pred)


# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter as a exogenous variables
# # ================================================================

def check_corr_cause_proxy_customer(hist_data_proxy_customers : Dict[Union[int,str],Customers], customer: Customers, input_features: Dict, number_of_proxy_customers: int):
    
    cus_rep = pd.concat([pd.DataFrame(hist_data_proxy_customers[i].data[input_features['Forecasted_param']]).rename(columns={input_features['Forecasted_param']: i}) for i in hist_data_proxy_customers.keys()],axis=1)
    
    corr = cus_rep.loc[input_features['Start training']:input_features['End training']].corrwith(customer.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']])
    corr.sort_values(ascending=False,inplace=True)

    corr = corr.head(min(3*number_of_proxy_customers,len(hist_data_proxy_customers.keys())))
    cus_rep = cus_rep[corr.to_dict()]

    try:
        for i in cus_rep.columns:
            if sm.tsa.stattools.grangercausalitytests(pd.concat([customer.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']], cus_rep[i].loc[input_features['Start training']:input_features['End training']]],axis=1
                                                                ), maxlag=1, verbose=False)[1][0]["ssr_ftest"][1] >= 0.05:
                cus_rep.drop(i,axis=1,inplace=True)   
                corr.drop(i,inplace=True) 

    except InfeasibleTestError:
        print(f'Warning: {customer.nmi} has values that are constant, or they follow an strange pattern that grangercausalitytests cannot be done on them!')

        sm.tools.sm_exceptions
    corr = corr.head(min(number_of_proxy_customers,len(cus_rep.columns)))
    cus_rep = cus_rep[corr.to_dict()]

    return cus_rep

def forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers: Dict[Union[int,str],Customers], customer: Customers, input_features: Dict, number_of_proxy_customers: Union[int, None] = None ) -> pd.DataFrame:
    """
    forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers,customer,input_features)

    This function generates forecasting values for a customer at using the some customers real-time measurements as proxies for a target customer.
    It uses the same the sk-forecast built in function that allows to use exogenous variables when forecasting a target customer. 
    More about this function can be found in "https://joaquinamatrodrigo.github.io/skforecast/0.4.3/notebooks/autoregresive-forecaster-exogenous.html".

    It requires three inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    And, the third input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """    
    
    # print(customer's nmi)
    print(" Customer nmi: {first}".format(first = customer.nmi))

    if number_of_proxy_customers is None:
        number_of_proxy_customers = min(10,len(hist_data_proxy_customers.keys()))

    customers_proxy = check_corr_cause_proxy_customer(hist_data_proxy_customers, customer, input_features, number_of_proxy_customers)

    customer.forecaster = ForecasterAutoreg(
            regressor = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.Ridge()),  
            lags      = input_features['Window size']     
        )

    customer.forecaster.fit(y    = customer.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']],
                            exog = customers_proxy.loc[input_features['Start training']:input_features['End training']])

    datetimes = customer.data.index

    check_time_zone_ = has_timezone(customer.data[input_features['Forecasted_param']].index[0])
    if check_time_zone_ == True:
        new_index =  pd.date_range(
                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ datetime.timedelta(days=input_features['Windows to be forecasted']),
                        freq=input_features['data_freq'],
                        tz="Australia/Sydney").delete(-1)
    else:
        new_index =  pd.date_range(
                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ datetime.timedelta(days=input_features['Windows to be forecasted']),
                        freq=input_features['data_freq']).delete(-1)

    customer.predictions_exog_rep = customer.forecaster.predict(steps = len(new_index),
                                                       last_window = customer.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']],
                                                       exog = customers_proxy.loc[new_index[0]:new_index[-1]] ).to_frame().set_index(new_index)
    

    result = customer.predictions_exog_rep
    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


def forecast_pointbased_exog_reposit_multiple_nodes_list_comprehension(hist_data_proxy_customers: Dict[Union[int,str],Customers], n_customers: Dict[Union[int,str],Customers], input_features: Dict, number_of_proxy_customers: Union[int, None] = None ) -> pd.DataFrame:

    preds = [forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers,n_customers[i],input_features, number_of_proxy_customers) for i in n_customers.keys()]

    return pd.concat(preds, axis=0)



def pool_executor_parallel_exog(function_name, hist_data_proxy_customers, repeat_iter, input_features, number_of_proxy_customers):
    '''
    pool_executor_parallel(function_name,repeat_iter,input_features)
    
    This function is used to parallelised the forecasting for each nmi
    '''

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,itertools.repeat(hist_data_proxy_customers),repeat_iter,itertools.repeat(input_features),itertools.repeat(number_of_proxy_customers)))  
    return results

    # with ThreadPoolExecutor(max_workers=input_features['core_usage']) as executor:
    #     results = list(executor.map(function_name,itertools.repeat(hist_data_proxy_customers),repeat_iter,itertools.repeat(input_features),itertools.repeat(number_of_proxy_customers)))  
    # return results


def forecast_pointbased_exog_reposit_multiple_nodes(hist_data_proxy_customers: Dict[Union[int,str],Customers], customers: Dict[Union[int,str],Customers], input_features: Dict, number_of_proxy_customers: Union[int, None] = None ) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    predictions_prallel = pool_executor_parallel_exog(forecast_pointbased_exog_reposit_single_node,hist_data_proxy_customers,customers.values(),input_features,number_of_proxy_customers)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel


# # ================================================================
# # Load_forecasting Using linear regression of time of day as a exogenous variables
# # ================================================================
def forecast_pointbased_exog_time_single_node(customer: Customers, input_features: Dict) -> pd.DataFrame:
    """
    forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers,customer,input_features)

    This function generates forecasting values for a customer at using the some customers real-time measurements as proxies for a target customer.
    It uses the same the sk-forecast built in function that allows to use exogenous variables when forecasting a target customer. 
    More about this function can be found in "https://joaquinamatrodrigo.github.io/skforecast/0.4.3/notebooks/autoregresive-forecaster-exogenous.html".

    It requires three inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    And, the third input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """    
    
    time_of_day = pd.DataFrame(customer.data.index.strftime("%H%M%S").astype(int),index=customer.data.index)
    time_of_day['weekday'] = (customer.data.index.day_of_week > 4).astype(int)

    customer.forecaster = ForecasterAutoreg(
            regressor = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.Ridge()),  
            lags      = input_features['Window size']     
        )

    customer.forecaster.fit(y    = customer.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']],
                            exog = time_of_day.loc[input_features['Start training']:input_features['End training']])

    check_time_zone_ = has_timezone(customer.data[input_features['Forecasted_param']].index[0])
    if check_time_zone_ == True:
        new_index =  pd.date_range(
                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (customer.data.index[1]-customer.data.index[0]),
                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (customer.data.index[1]-customer.data.index[0])+ datetime.timedelta(days=input_features['Windows to be forecasted']),
                        freq=input_features['data_freq'],
                        tz="Australia/Sydney").delete(-1)
    else:
        new_index =  pd.date_range(
                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (customer.data.index[1]-customer.data.index[0]),
                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (customer.data.index[1]-customer.data.index[0])+ datetime.timedelta(days=input_features['Windows to be forecasted']),
                        freq=input_features['data_freq']).delete(-1)

    customer.predictions_exog_rep = customer.forecaster.predict(steps = len(new_index),
                                                        last_window = customer.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']],
                                                        exog = time_of_day.loc[new_index[0]:new_index[-1]] ).to_frame().set_index(new_index)


    result = customer.predictions_exog_rep
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result


# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter as a exogenous variables
# # This function works on a data with two types, i.e., reposit and smart meters together
# # ================================================================

def forecast_mixed_type_customers(customers: Dict[Union[int,str],Customers], 
                                     participants: List[Union[int,str]], 
                                     input_features: Dict,
                                     end_participants_date: Union[datetime.datetime, pd.Timestamp, None] = None, 
                                     end_non_participants_date: Union[datetime.datetime, pd.Timestamp, None] = None,                                 
                                     non_participants:  Union[List[Union[int,str]], None] = None,
                                     number_of_proxy_customers: Union[int, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        

    # create a dictionary of customers with the participant nmi as keys.
    customers_partipant = {i: copy.deepcopy(customers[i]) for i in participants}

    # if the non-participants are not mentioned. All the customers that are not participating are considered as non-participant.
    if non_participants is None:
        non_participants = [i for i in customers.keys() if i not in participants]

    # if end_participants_date is not given, the last non-Nan value of the the first participant customer is used as the end_participants_date.
    if end_participants_date is None:
        end_participants_date = customers[participants[0]].data[input_features['Forecasted_param']].last_valid_index() 

    # if end_non_participants_date is not given, the last non-Nan value of the the first non-participant customer is used as the end_participants_date. 
    if end_non_participants_date is None:
        end_non_participants_date = customers[non_participants[0]].data[input_features['Forecasted_param']].last_valid_index()    

    # create a dictionary of customers with the non-participant nmi as keys.
    customers_non_participant = {i: customers[i] for i in non_participants}

    # generate forecate for participant customers.
    participants_pred = forecast_pointbased_autoregressive_multiple_nodes(customers_partipant, input_features)

    # combine forecast and historical data for participant customers.
    for i in participants_pred.index.levels[0]:
        temp = pd.concat([pd.DataFrame(customers_partipant[i].data[input_features['Forecasted_param']]),participants_pred.loc[i]])
        customers_partipant[i].data.drop(input_features['Forecasted_param'], axis=1, inplace = True)
        customers_partipant[i].data = pd.concat([customers_partipant[i].data,temp], axis=1)

    # update the inpute feature parameter, so that it matches the dates for the non-participant customers.
    input_features['End training'] = end_non_participants_date.strftime('%Y-%m-%d %H:%M:%S')
    input_features['Last-observed-window'] = end_non_participants_date.strftime('%Y-%m-%d %H:%M:%S')
    input_features['Windows to be forecasted'] = (end_participants_date - end_non_participants_date).days + input_features['Windows to be forecasted']

    if number_of_proxy_customers is None:
        number_of_proxy_customers = min(10,len(participants))

    # generate forecate for non-participant customers.
    non_participants_pred = forecast_pointbased_exog_reposit_multiple_nodes(customers_partipant, customers_non_participant, input_features, number_of_proxy_customers)

    return participants_pred, non_participants_pred


# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and time as a exogenous variables
# # ================================================================
def forecast_pointbased_exog_reposit_time_xgboost_single_node(hist_data_proxy_customers: Dict[Union[int,str],Customers], customer: Customers, input_features: Dict, number_of_proxy_customers: Union[int, None] = None ) -> pd.DataFrame:
    """
    forecast_pointbased_exog_reposit_single_node(hist_data_proxy_customers,customer,input_features)

    This function generates forecasting values for a customer at using the some customers real-time measurements as proxies for a target customer.
    It uses the same the sk-forecast built in function that allows to use exogenous variables when forecasting a target customer. 
    More about this function can be found in "https://joaquinamatrodrigo.github.io/skforecast/0.4.3/notebooks/autoregresive-forecaster-exogenous.html".

    It requires three inputs. The first input is a dictionry of known customers with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. 
    The second input is the target customer instance generated by the initilase function.
    And, the third input is the input_features which is a dictionary of input preferences generated by the initilase function.
    """    
    
    # print(customer's nmi)
    print(" Customer nmi: {first}".format(first = customer.nmi))

    if number_of_proxy_customers is None:
        number_of_proxy_customers = min(10,len(hist_data_proxy_customers.keys()))

    customers_proxy = check_corr_cause_proxy_customer(hist_data_proxy_customers, customer, input_features, number_of_proxy_customers)

    # exog_time = pd.DataFrame({'datetime': customers_proxy.index})
    # exog_time = exog_time.set_index(customers_proxy.index)
    # customers_proxy['minute_sin'] = np.sin(2 * np.pi * (exog_time.datetime.dt.hour*60 + exog_time.datetime.dt.minute) / 1440)
    # customers_proxy['minute_cos'] = np.cos(2 * np.pi * (exog_time.datetime.dt.hour*60 + exog_time.datetime.dt.minute) / 1440)

    customer.forecaster = ForecasterAutoreg(
            regressor = xgboost.XGBRegressor(),  
            lags      = input_features['Window size']     
        )

    customer.forecaster.fit(y    = customer.data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']],
                            exog = customers_proxy.loc[input_features['Start training']:input_features['End training']])

    datetimes = customer.data.index

    check_time_zone_ = has_timezone(customer.data[input_features['Forecasted_param']].index[0])
    if check_time_zone_ == True:
        new_index =  pd.date_range(
                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ datetime.timedelta(days=input_features['Windows to be forecasted']),
                        freq=input_features['data_freq'],
                        tz="Australia/Sydney").delete(-1)
    else:
        new_index =  pd.date_range(
                        start=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0]),
                        end=datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") + (datetimes[1]-datetimes[0])+ datetime.timedelta(days=input_features['Windows to be forecasted']),
                        freq=input_features['data_freq']).delete(-1)

    customer.predictions_exog_rep = customer.forecaster.predict(steps = len(new_index),
                                                       last_window = customer.data[input_features['Forecasted_param']].loc[(datetime.datetime.strptime(input_features['Last-observed-window'],"%Y-%m-%d %H:%M:%S") - datetime.timedelta(days=input_features['Windows to be forecasted'])).strftime("%Y-%m-%d %H:%M:%S"):input_features['Last-observed-window']],
                                                       exog = customers_proxy.loc[new_index[0]:new_index[-1]] ).to_frame().set_index(new_index)
    

    result = customer.predictions_exog_rep
    result.rename(columns={'pred': input_features['Forecasted_param']}, inplace = True)
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'datetime'}, inplace = True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return result

def forecast_pointbased_exog_reposit_time_xgboost_multiple_nodes(hist_data_proxy_customers: Dict[Union[int,str],Customers], customers: Dict[Union[int,str],Customers], input_features: Dict, number_of_proxy_customers: Union[int, None] = None ) -> pd.DataFrame:
    """
    forecast_pointbased_autoregressive_multiple_nodes(customers_nmi,input_features)

    This function generates forecasting values multiple customer using the autoregressive recursive multi-step forecasting method.
    It requires two inputs. The first input is a dictionry with keys being customers' nmi and values being their asscoated Customer instance generated by the initilase function. The second input is the input_features which is a dictionary 
    of input preferences generated by the initilase function.
    """

    predictions_prallel = pool_executor_parallel_exog(forecast_pointbased_exog_reposit_time_xgboost_single_node,hist_data_proxy_customers,customers.values(),input_features,number_of_proxy_customers)
 
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return predictions_prallel