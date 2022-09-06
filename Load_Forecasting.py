# Libraries
# ==============================================================================
# General
import pandas as pd

# Modelling and Forecasting
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster

# Get data from the ReadData script
from ReadData import data, customers_nmi

# Warnings configuration
import warnings
warnings.filterwarnings('ignore')

# Define a class for all the nmis and load forecating functions
# ==============================================================================
class customers_class:
     
    Instances = [] 
    customer_predictions = []

    def __init__(self, nmi):

        
        self.nmi = nmi      # store nmi in each object              
        
        self.data = data.loc[self.nmi].copy()      # store data in each object 
        self.end_validation = self.data.last('3D').index[0] - pd.Timedelta(minutes=1)      # 3D means 3 days. This number is used as the number of days to be predicted
        self.end_train = self.data.last('6D').index[0] - pd.Timedelta(minutes=1)      # 6D means 6 days. Different between this number and the above number is used to express the number of days to be used for validation
        self.data_train = self.data.loc[: self.end_train, :]
        self.data_val   = self.data.loc[self.end_train:self.end_validation, :]
        self.data_test  = self.data.loc[self.end_validation:, :]
        
        customers_class.Instances.append(self)
    
    # This function outputs the forecasts using a Recursive multi-step point-forecasting method of each nmi individually
    def Backtest(self):

        # Create and train forecaster
        # ==============================================================================
        self.forecaster = ForecasterAutoreg(
                        regressor = make_pipeline(StandardScaler(), Ridge()),
                        lags      = 48      # The used data set has a 30-minute resolution. So, 48 denotes one full day window
                    )
        self.forecaster.fit(y=self.data.loc[:self.end_validation, 'active_power'])

        # Backtest
        # ==============================================================================
        self.metric,self.predictions = backtesting_forecaster(
                                    forecaster = self.forecaster,
                                    y          = self.data.active_power,
                                    initial_train_size = len(self.data.loc[:self.end_validation]),
                                    # fixed_train_size   = False,
                                    steps      = 48,
                                    metric     = 'mean_absolute_error',
                                    refit      = False,
                                    verbose    = True
                            )
        self.predictions = self.predictions.set_index(self.data_test.index)
        return self.predictions, self.metric    

    # This function outputs the forecasts using a Recursive multi-step probabilistic forecasting method of each nmi individually
    def Interval_backtest(self):

        # Create and train forecaster
        # ==============================================================================
        self.forecaster = ForecasterAutoreg(
                        regressor = make_pipeline(StandardScaler(), Ridge()),
                        lags      = 48      # The used data set has a 30-minute resolution. So, 48 denotes one full day window
                    )
        self.forecaster.fit(y=self.data.loc[:self.end_validation, 'active_power'])


        # Backtesting
        # ==============================================================================
        self.metric,self.predictions = backtesting_forecaster(
                                    forecaster = self.forecaster,
                                    y          = self.data.active_power,
                                    initial_train_size = len(self.data.loc[:self.end_validation]),
                                    steps      = 48,
                                    refit      = True,
                                    interval   = [10, 90],
                                    n_boot     = 1000,
                                    metric     = 'mean_squared_error',
                                    verbose    = False
                                )

        self.predictions = self.predictions.set_index(self.data_test.index)
        return self.predictions, self.metric 


    # This function outputs the forecasts all nmis
    @classmethod
    def Backtest_all(cls):
        for Instance in cls.Instances:

            # Create and train forecaster
            # ==============================================================================
            Instance.forecaster = ForecasterAutoreg(
                            regressor = make_pipeline(StandardScaler(), Ridge()),
                            lags      = 48
                        )
            Instance.forecaster.fit(y=Instance.data.loc[:Instance.end_validation, 'active_power'])

            # Backtest
            # ==============================================================================
            Instance.metric,Instance.predictions = backtesting_forecaster(
                                        forecaster = Instance.forecaster,
                                        y          = Instance.data.active_power,
                                        initial_train_size = len(Instance.data.loc[:Instance.end_validation]),
                                        # fixed_train_size   = False,
                                        steps      = 48,
                                        metric     = 'mean_absolute_error',
                                        refit      = False,
                                        verbose    = True
                                )
            Instance.predictions = Instance.predictions.set_index(Instance.data_test.index)
            
            cls.customer_predictions.append(Instance.predictions.rename(columns={'pred': str(Instance.nmi)}))

# Create an instance for each nmi in the customers_class class
customers = {}
for customer in customers_nmi:
    customers[customer] = customers_class(customer)

