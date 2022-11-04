import matplotlib.pyplot as plt
from copy import deepcopy as copy
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np
from sklearn.metrics import mean_squared_error


from converge_load_forecasting import read_data
from converge_load_forecasting import SDD_min_solar_single_node,SDD_Same_Irrad_multiple_times,SDD_Same_Irrad_no_PV_houses_multiple_times,SDD_constant_PF_single_node,SDD_known_pvs_single_node,SDD_using_temp_single_node,SDD_known_pvs_temp_single_node_algorithm

# input_features = {  'file_type': 'Converge',
#                     'file_name': '_WANNIA_8MB_MURESK-nmi-loads.csv',
#                     'nmi_type_name': 'nmi.csv',
#                     'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
#                     'Start training': '2022-07-01',
#                     'End training': '2022-07-27',
#                     'Last-observed-window': '2022-07-27',
#                     'Window size': 48 ,
#                     'Windows to be forecasted':    3,     
#                     'data_freq' : '30T',
#                     'core_usage': 8      
#                      }

input_features = {  'file_type': 'NextGen',
                    'file_name': 'NextGen.csv',
                    'Forecasted_param': 'active_power',         # set this parameter to the value that is supposed to be forecasted. Acceptable: 'active_power' or 'reactive_power'
                    'Start training': '2018-01-01',
                    'End training': '2018-02-01',
                    'Last-observed-window': '2018-02-01',
                    'Window size':  288,
                    'Windows to be forecasted':    3,
                    'data_freq' : '5T',
                    'core_usage': 8      }  

data, customers_nmi,customers_nmi_with_pv,datetimes, customers,data_weather = read_data(input_features)

# Set this value to choose an nmi from customers_nmi 
# Examples
# nmi = customers_nmi[10]
nmi = customers_nmi_with_pv[1]
# Dates_for_plot = '2018-01-10':'2018-01-13'
####################################################################
## 7 different techniques to disaggregate solar and demand 
####################################################################

################
## technique 1
################

pv1 = SDD_min_solar_single_node(customers[nmi],input_features)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv1.loc[nmi].pv_disagg['2018-01-10':'2018-01-12'].plot(label = 'min pos PV')
customers[nmi].data.pv['2018-01-10':'2018-01-12'].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ###############
# ## technique 2
# ################

pv2 = SDD_Same_Irrad_multiple_times(data,input_features,customers[customers_nmi[0]].data['2018-01-10':'2018-01-13'].index,customers_nmi_with_pv)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv2.loc[nmi].pv_disagg['2018-01-10':'2018-01-12'].plot(label = 'min pos PV')
customers[nmi].data.pv['2018-01-10':'2018-01-12'].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ################
# ## technique 3
# ################

# customers_without_pv = [customers_nmi_with_pv[i] for i in np.random.randint(2, size=10)]
customers_without_pv  = [customers_nmi_with_pv[i] for i in [0,4,5,8,10,19,20,22,40,60]]  # randomly select 10 nmi as nmi's without pv
customers_with_pv = [i for i in customers_nmi_with_pv if i not in customers_without_pv]
pv3  = SDD_Same_Irrad_no_PV_houses_multiple_times(data,input_features,customers[customers_nmi[0]].data['2018-01-10':'2018-01-13'].index,customers_with_pv,customers_without_pv)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv3.loc[nmi].pv_disagg['2018-01-10':'2018-01-12'].plot(label = 'min pos PV')
customers[nmi].data.pv['2018-01-10':'2018-01-12'].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ################
# ## technique 4
# ################

pv4 = SDD_constant_PF_single_node(customers[nmi],input_features)


# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv4.loc[nmi].pv_disagg['2018-01-10':'2018-01-12'].plot(label = 'min pos PV')
customers[nmi].data.pv['2018-01-10':'2018-01-12'].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ################
# ## technique 5
# ################
known_pv_nmis = [customers_nmi_with_pv[i] for i in [5,10,30,40]]
customers_known_pv = {i: customers[i] for i in known_pv_nmis}
# from more_itertools import take
# n_customers = dict(take(4, customers.items())) 
# a = Generate_disaggregation_using_knownPVS_all(n_customers,input_features,customers_known_pv,datetimes)
pv5 = SDD_known_pvs_single_node(customers[nmi],customers_known_pv,datetimes)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv5.pv_disagg['2018-01-10':'2018-01-12'].plot(label = 'min pos PV')
customers[nmi].data.pv['2018-01-10':'2018-01-12'].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ################
# ## technique 6
# ################
pv6 = SDD_using_temp_single_node(customers[nmi],data_weather)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv6.pv_disagg['2018-01-10':'2018-01-12'].plot(label = 'min pos PV')
customers[nmi].data.pv['2018-01-10':'2018-01-12'].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()

# ################
# ## technique 7
# ################
customers_known_pv = {customers_nmi_with_pv[i]: customers[customers_nmi_with_pv[i]] for i in [10,40,50]}
pv7 = SDD_known_pvs_temp_single_node_algorithm(customers[nmi],data_weather,customers_known_pv,datetimes)


# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv7.pv_disagg['2018-01-10':'2018-01-12'].plot(label = 'min pos PV')
customers[nmi].data.pv['2018-01-10':'2018-01-12'].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()




