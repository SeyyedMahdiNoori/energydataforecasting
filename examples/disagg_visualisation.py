import matplotlib.pyplot as plt
from copy import deepcopy as copy
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


import converge_load_forecasting as cld

# Donwload if data is availbale in csv format
url_data = 'https://raw.githubusercontent.com/SeyyedMahdiNoori/converge_load_forecasting_data/main/NextGen_example.csv'
raw_data = pd.read_csv(url_data, sep=',')

url_wather = 'https://raw.githubusercontent.com/SeyyedMahdiNoori/converge_load_forecasting_data/main/Canberra_weather_data.csv'
weather_data = pd.read_csv(url_wather, sep=',')

data_initialised = cld.initialise(raw_data=raw_data,
                                    raw_proxy_data= weather_data,
                                    core_usage = 4
                              )

customers_nmi_with_pv = data_initialised.customers_nmi

# ################
# ## Initialise variables 
# ################
# # Set this value to choose an nmi from customers_nmi 
# # Examples
nmi = '14-id_292'

# Dates for disaggregation
Dates_for_plot_start = '2018-12-23'
Dates_for_plot_end = '2018-12-23'
time_steps_for_disagg = data_initialised.customers[data_initialised.customers_nmi[0]].data[Dates_for_plot_start:Dates_for_plot_end].index

# Required variables for some of the techniques
customers_without_pv_nmi  = [customers_nmi_with_pv[i] for i in np.random.default_rng().choice(len(customers_nmi_with_pv), size=30, replace=False) if i != nmi]  # randomly select 10 nmi as nmi's without pv
customers_with_pv_nmi = [i for i in customers_nmi_with_pv if i not in customers_without_pv_nmi]
known_pv_nmi = [customers_with_pv_nmi[i] for i in np.random.default_rng().choice(len(customers_with_pv_nmi), size=3, replace=False) if i != nmi]
customers_known_pv = {i: data_initialised.customers[i] for i in known_pv_nmi}


# ####################################################################
# ## 7 different techniques to disaggregate solar and demand 
# ####################################################################

# ################
# ## technique 1
# ################

pv1 = cld.SDD_min_solar_single_node(data_initialised.customers[nmi],data_initialised.input_features)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv1.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Minimum Solar Generation')
data_initialised.customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ###############
# ## technique 2
# ################

pv2 = cld.SDD_Same_Irrad_multiple_times(time_steps_for_disagg,data_initialised.customers,data_initialised.input_features)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv2.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Same Irradiance')
data_initialised.customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ################
# ## technique 3
# ################
pv3  = cld.SDD_Same_Irrad_no_PV_houses_multiple_times(time_steps_for_disagg,data_initialised.customers,customers_with_pv_nmi,customers_without_pv_nmi,data_initialised.input_features)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv3.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Same Irradiance and Houses Without PV Installation')
data_initialised.customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# ################
# ## technique 4
# ################
pv4 = cld.SDD_constant_PF_single_node(data_initialised.customers[nmi],data_initialised.input_features)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv4.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Constant Power Factor Demand')
data_initialised.customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
plt.show()


# ################
# ## technique 5
# ################
pv5 = cld.SDD_known_pvs_single_node(data_initialised.customers[nmi],customers_known_pv,time_steps_for_disagg)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv5.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Measurements from Neighbouring Sites')
data_initialised.customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


# # ################
# # ## technique 6
# # ################

pv6 = cld.SDD_using_temp_single_node(data_initialised.customers[nmi],time_steps_for_disagg,data_initialised.data_proxy,data_initialised.input_features)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv6.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Weather Data')
data_initialised.customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()


################
## technique 7
################
pv7 = cld.SDD_known_pvs_temp_single_node_algorithm(data_initialised.customers[nmi],time_steps_for_disagg,data_initialised.data_proxy,data_initialised.input_features,customers_known_pv)

# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4.5))
pv7.loc[nmi].pv_disagg[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'Proxy Measurements from Neighbouring Sites and Weather Data')
data_initialised.customers[nmi].data.pv[Dates_for_plot_start:Dates_for_plot_end].plot(label = 'real')
plt.xlabel("Date")
plt.ylabel("Active Power (Watt)")
ax.set_title('Solar demand disaggregation')
ax.legend()
# plt.savefig('Active_power_data.eps', format='eps')
plt.show()
