import matplotlib.pyplot as plt
import copy
import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np
from sklearn.metrics import mean_squared_error


import converge_load_forecasting as clf 

# # Donwload if data is availbale in csv format
customersdatapath = './NextGen_example.csv'

data_initialised = clf.initialise(customersdatapath = customersdatapath,
                              forecasted_param = 'active_power',
                              end_training='2018-12-29',
                              last_observed_window='2018-12-29',
                              regressor = 'LinearRegression',
                              algorithm = 'iterated',
                              time_proxy = False,
                              days_to_be_forecasted=1)

# An arbitrary customer nmi to be use as target customer for forecasting
nmi = data_initialised.customers_nmi[10]
customer = {i: data_initialised.customers[nmi] for i in [nmi]}

# n number of customers (here arbitrarily 5 is chosen) to be forecasted parallely
n_customers = {data_initialised.customers_nmi[i]: data_initialised.customers[data_initialised.customers_nmi[i]] for i in np.random.default_rng(seed=1).choice(len(data_initialised.customers_nmi), size=5, replace=False)}

# n number of customers (here arbitrarily 5 is chosen) with know real-time values
hist_data_proxy_customers = {data_initialised.customers_nmi[i]: data_initialised.customers[data_initialised.customers_nmi[i]] for i in np.random.default_rng(seed=3).choice(len(data_initialised.customers_nmi), size=5, replace=False) if i not in n_customers.keys()}

# # ============================================================================
# # ============================================================================
# #     Data Visualisation
# # ============================================================================
# # ============================================================================

# # ===================================
# #     Time series plot
# # ===================================
fig, ax = plt.subplots(figsize=(12, 4.5))
data_initialised.customers[nmi].data.loc[data_initialised.input_features['start_training']:data_initialised.input_features['end_training']][data_initialised.input_features['Forecasted_param']].plot(ax=ax, label='train', linewidth=1)
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()

# # ===================================
# #     Zooming time series chart
# # ===================================
from datetime import timedelta
zoom = (data_initialised.datetimes[0] + timedelta(days = 1) ,data_initialised.datetimes[0] + timedelta(days = 2) )

fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)

main_ax = fig.add_subplot(grid[1:3, :])
zoom_ax = fig.add_subplot(grid[5:, :])

data_initialised.customers[nmi].data[data_initialised.input_features['Forecasted_param']].plot(ax=main_ax, c='black', alpha=0.5, linewidth=0.5)
min_y = min(data_initialised.customers[nmi].data[data_initialised.input_features['Forecasted_param']])
max_y = max(data_initialised.customers[nmi].data[data_initialised.input_features['Forecasted_param']])
main_ax.fill_between(zoom, min_y, max_y, facecolor='blue', alpha=0.5, zorder=0)
main_ax.set_xlabel('')

data_initialised.customers[nmi].data.loc[zoom[0]: zoom[1]][data_initialised.input_features['Forecasted_param']].plot(ax=zoom_ax, color='blue', linewidth=2)

main_ax.set_title(f'Electricity active_power: {data_initialised.customers[nmi].data.index.min()}, {data_initialised.customers[nmi].data.index.max()}', fontsize=14)
zoom_ax.set_title(f'Electricity active_power: {zoom}', fontsize=14)
plt.subplots_adjust(hspace=1)
plt.show()


# # ===================================
# #     Boxplot for weekly seasonality
# # ===================================
fig, ax = plt.subplots(figsize=(9, 4))
data_initialised.customers[nmi].data['week_day'] = data_initialised.customers[nmi].data.index.day_of_week + 1
data_initialised.customers[nmi].data.boxplot(column='active_power', by='week_day', ax=ax)
data_initialised.customers[nmi].data.groupby('week_day')['active_power'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('Demand')
ax.set_title('Demand distribution by week day')
ax.set_title('')
plt.xlabel("Day of Week")
# fig.suptitle('')
plt.ylabel("Active Power (Watt)")
# plt.savefig('Seasonality_Week.eps', format='eps')
plt.show()

# # ===================================
# #     Boxplot for daily seasonality
# # ===================================
fig, ax = plt.subplots(figsize=(9, 4))
data_initialised.customers[nmi].data['hour_day'] = data_initialised.customers[nmi].data.index.hour + 1
data_initialised.customers[nmi].data.boxplot(column='active_power', by='hour_day', ax=ax)
data_initialised.customers[nmi].data.groupby('hour_day')['active_power'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('Demand')
plt.xlabel("Hour of day")
plt.ylabel("Active Power (Watt)")
ax.set_title('')
ax.set_title('Demand distribution by the time of the day')
# fig.suptitle('')
# plt.savefig('Seasonality_Day.eps', format='eps')
plt.show()

# # ===================================
# #     Autocorrelation plot
# # ===================================
fig, ax = plt.subplots(figsize=(7, 3))
plot_acf(data_initialised.customers[nmi].data[data_initialised.input_features['Forecasted_param']], ax=ax, lags=data_initialised.input_features['window_size'])
plt.show()

# # ===================================
# #     Partial autocorrelation plot
# # ===================================
fig, ax = plt.subplots(figsize=(9, 4.5))
plot_pacf(data_initialised.customers[nmi].data[data_initialised.input_features['Forecasted_param']], ax=ax, lags=data_initialised.input_features['window_size'])
plt.xlabel("Lages")
plt.ylabel("PACF")
# ax.set_title('')
# plt.savefig('Partial_autocorrelation.eps', format='eps')
plt.show()


# # ====================================================================================================
# # Iterated multi-step point-based forecasting method with LinearRegression and Ridge
# # ====================================================================================================

# generate forecasting values for a specific nmi using a recursive multi-step point-forecasting method
prediction = clf.forecast_pointbased_multiple_nodes(customer,data_initialised.input_features)

fig, ax = plt.subplots(figsize=(12, 3.5))
prediction.loc[nmi][data_initialised.input_features['Forecasted_param']].plot(ax=ax,linewidth=2,label='prediction-iterated-LR')
customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[prediction.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (kW)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show() 


# # ====================================================================================================
# # Iterated multi-step point-based forecasting method with LinearRegression and Ridge, and using Time as exogenous
# # ====================================================================================================

# generate forecasting values for a specific nmi using a recursive multi-step point-forecasting method
data_initialised.input_features['exog'] = True
prediction = clf.forecast_pointbased_multiple_nodes(customer,data_initialised.input_features)

fig, ax = plt.subplots(figsize=(12, 3.5))
prediction.loc[nmi][data_initialised.input_features['Forecasted_param']].plot(ax=ax,linewidth=2,label='prediction-iterated-LR-time')
customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[prediction.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (kW)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show() 

# # ====================================================================================================
# # Iterated multi-step point-based forecasting method with XGBoost, and using Time as exogenous 
# # ====================================================================================================

# generate forecasting values for a specific nmi using a recursive multi-step point-forecasting method
data_initialised.input_features['regressor'] = clf.select_regressor('XGBoost')
prediction = clf.forecast_pointbased_multiple_nodes(customer,data_initialised.input_features)

fig, ax = plt.subplots(figsize=(12, 3.5))
prediction.loc[nmi][data_initialised.input_features['Forecasted_param']].plot(ax=ax,linewidth=2,label='prediction-iterated-XGBoost')
customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[prediction.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (kW)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show() 

# # ====================================================================================================
# # Direct multi-step point-based forecasting method with LinearRegression, and using Time as exogenous 
# # ====================================================================================================

# generate forecasting values for a specific nmi using a recursive multi-step point-forecasting method
data_initialised.input_features['regressor'] = clf.select_regressor('LinearRegression')
data_initialised.input_features['algorithm'] = 'direct'
prediction = clf.forecast_pointbased_multiple_nodes(customer,data_initialised.input_features)

fig, ax = plt.subplots(figsize=(12, 3.5))
prediction.loc[nmi][data_initialised.input_features['Forecasted_param']].plot(ax=ax,linewidth=2,label='prediction-direct-LR')
customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[prediction.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (kW)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show() 

# # ====================================================================================================
# # Rectified multi-step point-based forecasting method with LinearRegression, and using Time as exogenous 
# # ====================================================================================================

# generate forecasting values for a specific nmi using a recursive multi-step point-forecasting method
data_initialised.input_features['regressor'] = clf.select_regressor('LinearRegression')
data_initialised.input_features['algorithm'] = 'rectified'
prediction = clf.forecast_pointbased_multiple_nodes(customer,data_initialised.input_features)

fig, ax = plt.subplots(figsize=(12, 3.5))
prediction.loc[nmi][data_initialised.input_features['Forecasted_param']].plot(ax=ax,linewidth=2,label='prediction-rectified-LR')
customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[prediction.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (kW)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show() 


# # ====================================================================================================
# # Stacking multi-step point-based forecasting method with LinearRegression, and using Time as exogenous 
# # ====================================================================================================

# generate forecasting values for a specific nmi using a recursive multi-step point-forecasting method
data_initialised.input_features['regressor'] = clf.select_regressor('LinearRegression')
data_initialised.input_features['algorithm'] = 'stacking'
prediction = clf.forecast_pointbased_multiple_nodes(customer,data_initialised.input_features)

fig, ax = plt.subplots(figsize=(12, 3.5))
prediction.loc[nmi][data_initialised.input_features['Forecasted_param']].plot(ax=ax,linewidth=2,label='prediction-stacking-LR')
customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[prediction.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (kW)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show() 

# # ==================================================
# # Recursive multi-step probabilistic forecasting method
# # ==================================================

# generate forecasting values for a specific nmi using a recursive multi-step probabilistic forecasting method
data_initialised.input_features['algorithm'] = 'iterated'
prediction = clf.forecast_inetervalbased_multiple_nodes(customer,data_initialised.input_features)

import matplotlib.ticker as ticker
fig, ax = plt.subplots(figsize=(12, 3.5))
customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[prediction.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
ax.fill_between(
    prediction.loc[nmi].index,
    prediction.loc[nmi]['lower_bound'],
    prediction.loc[nmi]['upper_bound'],
    color = 'deepskyblue',
    alpha = 0.3,
    label = '80% interval'
)
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (kW)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show() 

# # # ================================================================
# # # Load_forecasting Using linear regression of Reposit data and smart meters
# # # ================================================================
res_rep_lin_single_time_single = clf.forecast_lin_reg_proxy_measures_single_node(hist_data_proxy_customers,customer[nmi],data_initialised.input_features)

fig, ax = plt.subplots(figsize=(12, 3.5))
res_rep_lin_single_time_single.loc[nmi][data_initialised.input_features['Forecasted_param']].plot(ax=ax,linewidth=2,label='repo_single_time')
customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[res_rep_lin_single_time_single.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (kW)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show() 


# # # ##
# # # Note: This approach requires more historical data for training and perform very poorly in cases of low data for training.
# # # Thus, it has not been depicted here.
# ##
# # # ================================================================
# # # Load_forecasting Using linear regression of Reposit data and smart meter, one for each time-step in a day
# # # ================================================================
# res_rep_lin_multi_time_single = clf.forecast_lin_reg_proxy_measures_separate_time_steps(hist_data_proxy_customers,customer[nmi],data_initialised.input_features)    

# fig, ax = plt.subplots(figsize=(12, 3.5))
# res_rep_lin_multi_time_single.loc[nmi][data_initialised.input_features['Forecasted_param']].plot(ax=ax,linewidth=2,label='repo_multi_time')
# customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[res_rep_lin_multi_time_single.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
# ax.set_title('Prediction vs real demand')
# ax.legend()
# plt.xlabel("Time")
# plt.ylabel("Active Power (kW)")
# plt.ylim((-10, 10))
# # plt.savefig('Real_vs_pred.eps', format='eps')
# plt.show() 

# # ================================================================
# # Load_forecasting Using linear regression of Reposit data and smart meter
# # ================================================================
_ , res_rep_exog = clf.forecast_mixed_type_customers(
                                    customers = data_initialised.customers,   # customers to be forecasted         
                                    participants = list(hist_data_proxy_customers.keys()), # list of participant customers' nmi
                                    non_participants = [nmi],  # list of non-participant customers nmi
                                    input_features = data_initialised.input_features, # input features generated by the initialise function
                                    )


fig, ax = plt.subplots(figsize=(12, 3.5))
res_rep_exog.loc[nmi][data_initialised.input_features['Forecasted_param']].plot(ax=ax,linewidth=2,label='repo_single_time-exogenous')
customer[nmi].data[data_initialised.input_features['Forecasted_param']].loc[res_rep_exog.index.levels[1]].plot(ax=ax, linewidth=2, label='real')
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (kW)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show() 
  