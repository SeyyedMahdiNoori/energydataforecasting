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
# # raw_data read from a server
# raw_data = pd.read_csv(NextGen_network_data_url)
# raw_weather_data = pd.read_csv(canberra_weather_url)
# data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather, input_features = initialise(raw_data = raw_data,raw_weather_data=raw_weather_data)

# Donwload if data is availbale in csv format
customersdatapath = './NextGen_example.csv'
weatherdatapath = './Canberra_weather_data.csv'
data, customers, input_features, customers_nmi, datetimes  = initialise(customersdatapath = customersdatapath,weatherdatapath = weatherdatapath)

import copy
customers_nmi_with_pv = copy.deepcopy(customers_nmi)

# ################
# ## Initialise variables 
# ################
# # Set this value to choose an nmi from customers_nmi 
# # Examples
nmi = customers_nmi_with_pv[10]

# Dates for disaggregation
Dates_for_plot_start = '2018-12-23'
Dates_for_plot_end = '2018-12-24'
time_steps_for_disagg = customers[customers_nmi[0]].data[Dates_for_plot_start:Dates_for_plot_end].index

# Required variables for some of the techniques
customers_without_pv  = [customers_nmi_with_pv[i] for i in np.random.default_rng().choice(len(customers_nmi_with_pv), size=30, replace=False) if i != nmi]  # randomly select 10 nmi as nmi's without pv
customers_with_pv = [i for i in customers_nmi_with_pv if i not in customers_without_pv]
known_pv_nmis = [customers_with_pv[i] for i in np.random.default_rng().choice(len(customers_with_pv), size=3, replace=False) if i != nmi]
customers_known_pv = {i: customers[i] for i in known_pv_nmis}


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
pv1.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Minimum Solar Generation')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ###############
# ## technique 2
# ################

pv2 = SDD_Same_Irrad_multiple_times(data,input_features,time_steps_for_disagg,customers_nmi_with_pv)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv2.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Same Irradiance')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ################
# ## technique 3
# ################

pv3  = SDD_Same_Irrad_no_PV_houses_multiple_times(data,input_features,time_steps_for_disagg,customers_with_pv,customers_without_pv)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv3.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Same Irradiance and Houses Without PV Installation')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
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
pv4.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Constant Power Factor Demand')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ################
# ## technique 5
# ################

pv5 = SDD_known_pvs_single_node(customers[nmi],customers_known_pv,time_steps_for_disagg)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv5.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Measurements from Neighbouring Sites')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ################
# ## technique 6
# ################

pv6 = SDD_using_temp_single_node(customers[nmi],time_steps_for_disagg,weatherdatapath=weatherdatapath)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv6.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Weather Data')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()

# ################
# ## technique 7
# ################
pv7 = SDD_known_pvs_temp_single_node_algorithm(customers[nmi],customers_known_pv,time_steps_for_disagg,weatherdatapath=weatherdatapath)


# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv7.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Proxy Measurements from Neighbouring Sites and Weather Data')
customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()
