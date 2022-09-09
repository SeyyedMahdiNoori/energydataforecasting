# Libraries
# ==============================================================================
# General
import pandas as pd
import numpy as np

# Modelling and Forecasting
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline # Construct a Pipeline from the given estimators (for more information: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance.
from skforecast.ForecasterAutoreg import ForecasterAutoreg # A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from datetime import date, timedelta


# Get data from the ReadData script
from ReadData import data, customers_nmi, input_features

# Warnings configuration
import warnings
warnings.filterwarnings('ignore')


# Define a class for all the nmis and load forecating functions
# ==============================================================================
class customers_class:

    # windowsize = input_features['Window size']
    # stepstoBeforecasted = input_features['Windows to be forecasted'] * windowsize
    # Start_training = input_features['Start training']
    # End_training = input_features['End training']
    # Last_observed_window = input_features['Last-observed-window']

    def __init__(self, nmi,input_features):

        self.nmi = nmi      # store nmi in each object              
        self.data = data.loc[self.nmi]      # store data in each object 
        self.data_train = self.data.loc[input_features['Start training']:input_features['End training']]

    # This function outputs the forecasts using a Recursive multi-step point-forecasting method of each nmi individually
    def Generate_forecaster_object(self,input_features):
        # Create a forecasting object
        self.forecaster = ForecasterAutoreg(
                regressor = make_pipeline(StandardScaler(), Ridge()),
                lags      = input_features['Window size']      # The used data set has a 30-minute resolution. So, 48 denotes one full day window
            )

        # Train the forecaster using the train data
        self.forecaster.fit(y=self.data_train.active_power)

    # This function optimises the parameters of the forecaster
    def Generate_optimised_forecaster_object(self,input_features):
        
        # These lines are used to hide the bar in the optimisation process
        from tqdm import tqdm
        from functools import partialmethod
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
                        y           = self.data_train.active_power,
                        param_grid  = param_grid,
                        # lags_grid   = lags_grid,
                        steps       =  48, # input_features['Window size'],
                        metric      = 'mean_absolute_error',
                        # refit       = False,
                        initial_train_size = len(self.data_train.active_power) - input_features['Window size'] * 10,
                        # fixed_train_size   = False,
                        return_best = True,
                        verbose     = False
                 )
        

    def Generate_prediction(self,input_features):
        # Generate predictions using normal forecasting
        Newindex = pd.date_range(start=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=1), end=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=input_features['Windows to be forecasted']+1),freq='30T').delete(-1)
        self.predictions = self.forecaster.predict(steps=input_features['Windows to be forecasted'] * input_features['Window size'], last_window=self.data.active_power.loc[input_features['Last-observed-window']]).to_frame().set_index(Newindex)

    def Generate_interval_prediction(self,input_features):
        # Generate predictions using normal forecasting

        # Create a time-index for the dates that are being predicted
        Newindex = pd.date_range(start=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=1), end=date(int(input_features['Last-observed-window'][0:4]),int(input_features['Last-observed-window'][5:7]),int(input_features['Last-observed-window'][8:10]))+timedelta(days=input_features['Windows to be forecasted']+1),freq='30T').delete(-1)
        
        # [10 90] considers 80% (90-10) confidence interval ------- n_boot: Number of bootstrapping iterations used to estimate prediction intervals.
        self.interval_predictions = self.forecaster.predict_interval(steps=input_features['Windows to be forecasted'] * input_features['Window size'], interval = [10, 90],n_boot = 1000, last_window=self.data.active_power.loc[input_features['Last-observed-window']]).set_index(Newindex)


# Create an instance for each nmi in the customers_class class
customers = {}
for customer in customers_nmi:
    customers[customer] = customers_class(customer,input_features)


