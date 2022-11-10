import matplotlib.pyplot as plt
from copy import deepcopy as copy
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


from converge_load_forecasting import initialise
from converge_load_forecasting import SDD_min_solar_single_node,SDD_Same_Irrad_multiple_times,SDD_Same_Irrad_no_PV_houses_multiple_times,SDD_constant_PF_single_node,SDD_known_pvs_single_node,SDD_using_temp_single_node,SDD_known_pvs_temp_single_node_algorithm

#### Use either path data approach if data is available or raw data approach to download it from a server

#### Use either path data approach if data is available or raw data approach to download it from a server
# # raw_data read from a server
NextGen_network_data_url = 'https://cloudstor.aarnet.edu.au/sender/download.php?token=087e5222-9919-4c67-af86-3e7d284e1ec2&files_ids=17805925'
raw_data = pd.read_csv(NextGen_network_data_url)
canberra_weather_url = 'https://cloudstor.aarnet.edu.au/sender/download.php?token=087e5222-9919-4c67-af86-3e7d284e1ec2&files_ids=17805920'
raw_weather_data = pd.read_csv(canberra_weather_url)
data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather, input_features = initialise(raw_data = raw_data,raw_weather_data=raw_weather_data)

# # Donwload if data is availbale in csv format
# customersdatapath = '/Users/mahdinoori/Documents/WorkFiles/Simulations/LoadForecasting/load_forecasting/data/Examples_data/NextGen_example.csv'
# weatherdatapath = '/Users/mahdinoori/Documents/WorkFiles/Simulations/LoadForecasting/load_forecasting/data/Examples_data/Canberra_weather_data.csv'
# data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather, input_features = initialise(customersdatapath = customersdatapath,weatherdatapath = weatherdatapath)




# # Set this value to choose an nmi from customers_nmi 
# # Examples
# # nmi = customers_nmi[10]
nmi = customers_nmi_with_pv[1]
Dates_for_plot_start = '2018-12-16'
Dates_for_plot_end = '2018-12-17'
# ####################################################################
# ## 7 different techniques to disaggregate solar and demand 
# ####################################################################

# ################
# ## technique 1
# ################

pv1 = SDD_min_solar_single_node(customers[nmi],input_features)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv1.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'min pos PV')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ###############
# ## technique 2
# ################

pv2 = SDD_Same_Irrad_multiple_times(data,input_features,customers[customers_nmi[0]].data[Dates_for_plot_start:Dates_for_plot_end].index,customers_nmi_with_pv)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv2.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'min pos PV')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
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
pv3  = SDD_Same_Irrad_no_PV_houses_multiple_times(data,input_features,customers[customers_nmi[0]].data[Dates_for_plot_start:Dates_for_plot_end].index,customers_with_pv,customers_without_pv)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv3.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'min pos PV')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
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
pv4.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'min pos PV')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
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
pv5.pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'min pos PV')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
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
pv6.pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'min pos PV')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
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
pv7.pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'min pos PV')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Behind the meter measurement')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()
