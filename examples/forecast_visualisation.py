import matplotlib.pyplot as plt
import copy
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np
from sklearn.metrics import mean_squared_error


from load_forecasting_functions import read_data

# # Set features of the predections
# input_features = {  'file_type': 'NextGen',
#                     'file_name': 'NextGen.csv',
#                     'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
#                     'Start training': '2018-01-01',
#                     'End training': '2018-02-01',
#                     'Last-observed-window': '2018-02-01',
#                     'Window size':  288,
#                     'Windows to be forecasted':    3,
#                     'data_freq' : '5T',
#                     'core_usage': 8      }  

input_features = {  'file_type': 'Converge',
                    'file_name': '_WANNIA_8MB_MURESK-nmi-loads.csv',
                    'nmi_type_name': 'nmi.csv',
                    'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
                    'Start training': '2022-07-01',
                    'End training': '2022-07-27',
                    'Last-observed-window': '2022-07-27',
                    'Window size': 48 ,
                    'Windows to be forecasted':    3,     
                    'data_freq' : '30T',
                    'core_usage': 8      
                     }


data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather = read_data(input_features)

# Set this value to choose an nmi from customers_nmi 
# Examples
# nmi = customers_nmi[10]
nmi = customers_nmi_with_pv[1]

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
customers[nmi].data.loc[input_features['Start training']:input_features['End training']][input_features['Forecasted_param']].plot(ax=ax, label='train', linewidth=1)
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()



# Zooming time series chart
# ==============================================================================
zoom = (datetimes[int(len(datetimes)/15)] ,datetimes[int(len(datetimes)/10)] )

fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)

main_ax = fig.add_subplot(grid[1:3, :])
zoom_ax = fig.add_subplot(grid[5:, :])

customers[nmi].data[input_features['Forecasted_param']].plot(ax=main_ax, c='black', alpha=0.5, linewidth=0.5)
min_y = min(customers[nmi].data[input_features['Forecasted_param']])
max_y = max(customers[nmi].data[input_features['Forecasted_param']])
main_ax.fill_between(zoom, min_y, max_y, facecolor='blue', alpha=0.5, zorder=0)
main_ax.set_xlabel('')

customers[nmi].data.loc[zoom[0]: zoom[1]][input_features['Forecasted_param']].plot(ax=zoom_ax, color='blue', linewidth=2)

main_ax.set_title(f'Electricity active_power: {customers[nmi].data.index.min()}, {customers[nmi].data.index.max()}', fontsize=14)
zoom_ax.set_title(f'Electricity active_power: {zoom}', fontsize=14)
plt.subplots_adjust(hspace=1)
plt.show()



# Boxplot for weekly seasonality
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
customers[nmi].data['week_day'] = customers[nmi].data.index.day_of_week + 1
customers[nmi].data.boxplot(column='active_power', by='week_day', ax=ax)
customers[nmi].data.groupby('week_day')['active_power'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('Demand')
ax.set_title('Demand distribution by week day')
ax.set_title('')
plt.xlabel("Day of Week")
# fig.suptitle('')
plt.ylabel("Active Power (Watt)")
# plt.savefig('Seasonality_Week.eps', format='eps')
plt.show()

# Boxplot for daily seasonality
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
customers[nmi].data['hour_day'] = customers[nmi].data.index.hour + 1
customers[nmi].data.boxplot(column='active_power', by='hour_day', ax=ax)
customers[nmi].data.groupby('hour_day')['active_power'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('Demand')
plt.xlabel("Hour of day")
plt.ylabel("Active Power (Watt)")
ax.set_title('')
ax.set_title('Demand distribution by the time of the day')
# fig.suptitle('')
# plt.savefig('Seasonality_Day.eps', format='eps')
plt.show()


# # # #### Not done yet
# # # # Violinplot
# # # # ==============================================================================
# # # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3.5))
# # # sns.violinplot(
# # #     x       = 'Demand',
# # #     y       = 'Holiday',
# # #     data    = data.assign(Holiday = data.Holiday.astype(str)),
# # #     palette = 'tab10',
# # #     ax      = ax
# # # )
# # # ax.set_title('Distribution of demand between holidays and non-holidays')
# # # ax.set_xlabel('Demand')
# # # ax.set_ylabel('Holiday');
# # # plt.show()

# Autocorrelation plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3))
plot_acf(customers[nmi].data[input_features['Forecasted_param']], ax=ax, lags=120)
plt.show()

# Partial autocorrelation plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4.5))
plot_pacf(customers[nmi].data[input_features['Forecasted_param']], ax=ax, lags=48*2)
plt.xlabel("Lages")
plt.ylabel("PACF")
# ax.set_title('')
# plt.savefig('Partial_autocorrelation.eps', format='eps')
plt.show()



# Plot Predition vs Real data using point-based approach
# ==============================================================================

customers[nmi].generate_forecaster(input_features)
customers[nmi].generate_prediction(input_features)
predictions= customers[nmi].predictions
fig, ax = plt.subplots(figsize=(12, 4.5))
customers[nmi].data[input_features['Forecasted_param']].loc[predictions.index.strftime('%m/%d/%Y').min():predictions.index.strftime('%m/%d/%Y').max()].plot(ax=ax, linewidth=2, label='real')
predictions.pred.plot(linewidth=2, label='prediction', ax=ax)
ax.set_title('Prediction vs real demand')
ax.legend()
plt.xlabel("Time")
plt.ylabel("Active Power (Watt)")
# plt.savefig('Real_vs_pred.eps', format='eps')
plt.show()   


# Plot Predition vs Real data using point-based approach with and without forecaster optimiser
# ==============================================================================

customers[nmi].generate_forecaster(input_features)
customers[nmi].generate_prediction(input_features)
predictions= customers[nmi].predictions

customers[nmi].generate_optimised_forecaster_object(input_features)
customers[nmi].generate_prediction(input_features)
predictions_optimised= customers[nmi].predictions

fig, ax = plt.subplots(figsize=(12, 3.5))
customers[nmi].data[input_features['Forecasted_param']].loc[predictions.index.strftime('%m/%d/%Y').min():predictions.index.strftime('%m/%d/%Y').max()].plot(ax=ax, linewidth=2, label='real')
predictions.pred.plot(linewidth=2, label='prediction', ax=ax)
predictions_optimised.pred.plot(linewidth=2, label='prediction-optimised', ax=ax)

y_true = np.array(list(customers[nmi].data[input_features['Forecasted_param']].loc[predictions.index.strftime('%m/%d/%Y').min():predictions.index.strftime('%m/%d/%Y').max()]))
y_pred = np.array(list(predictions.pred))
y_pred_optimised = np.array(list(predictions_optimised.pred))
mses = ((y_true-y_pred)**2).mean()
mses_optimised = ((y_true-y_pred_optimised)**2).mean()
print(f"error based (mse): {mses}")
print(f"error optimised (mse): {mses_optimised}")

ax.set_title('Prediction vs real demand')
ax.legend()
plt.show()   


# # Plot Interval Predition vs Real data using the Recursive multi-step probabilistic forecasting method
# # ==============================================================================
import matplotlib.ticker as ticker

customers[nmi].generate_forecaster(input_features)
customers[nmi].generate_interval_prediction(input_features)
predictions_interval= customers[nmi].interval_predictions

fig, ax=plt.subplots(figsize=(11, 4.5))
customers[nmi].data[input_features['Forecasted_param']].loc[predictions_interval.index.strftime('%m/%d/%Y').min():predictions_interval.index.strftime('%m/%d/%Y').max()].plot(ax=ax, linewidth=2, label='real')
ax.fill_between(
    predictions_interval.index,
    predictions_interval['lower_bound'],
    predictions_interval['upper_bound'],
    color = 'deepskyblue',
    alpha = 0.3,
    label = '80% interval'
)

ax.yaxis.set_major_formatter(ticker.EngFormatter())
plt.xlabel("Time")
plt.ylabel("Active Power (Watt)")
ax.set_title('Energy demand forecast')
ax.legend()
# plt.savefig('Real_vs_pred_interval.eps', format='eps')
plt.show()   


# # Plot Predition vs Real data using point-based approach
# # ==============================================================================
# from Load_Forecasting import generate_disaggregation_regression
# generate_disaggregation_regression()

# customers[nmi].generate_forecaster(input_features)
# customers[nmi].generate_prediction(input_features)
# predictions= customers[nmi].predictions

# input_features1 = copy.deepcopy(input_features)
# input_features1['Forecasted_param']= 'pv_disagg'
# customers[nmi].generate_forecaster(input_features1)
# customers[nmi].generate_prediction(input_features1)
# predictions_pv = customers[nmi].predictions

# input_features1['Forecasted_param']= 'demand_disagg'
# customers[nmi].generate_forecaster(input_features1)
# customers[nmi].generate_prediction(input_features1)
# predictions_demand = customers[nmi].predictions

# predictions_agg = predictions_demand + predictions_pv

# fig, ax = plt.subplots(figsize=(12, 4.5))
# customers[nmi].data[input_features['Forecasted_param']].loc[predictions.index.strftime('%m/%d/%Y').min():predictions.index.strftime('%m/%d/%Y').max()].plot(ax=ax, linewidth=2, label='real')
# predictions.pred.plot(linewidth=2, label='pred direct', ax=ax)
# predictions_agg.pred.plot(linewidth=2, label='pred disagg', ax=ax)
# # ax.set_title('Predictions vs real demand')
# ax.legend()
# plt.xlabel("Time")
# plt.ylabel("Active Power (Watt)")
# # plt.savefig('Real_vs_pred_diss.eps', format='eps')
# plt.show()   
