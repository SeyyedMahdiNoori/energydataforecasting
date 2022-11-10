# # ==================================================================================================
# # This package is developed as a part of the Converge project at the Australian National University
# # Users can use this package to generate forecasting values for electricity demand. It can also be 
# # used to disaggregate solar and demand from the aggregated measurement at the connection point
# # to the system
# # Two method are implemented to generate forecasting values:
# # 1. Recursive multi-step point-forecasting method
# # 2. Recursive multi-step probabilistic forecasting method
# # For further details on the disaggregation algorithms, please refer to the user manual file
# #
# # Below is an example of the main functionalities in this package
# # ==================================================================================================


from more_itertools import take
import pandas as pd
import numpy as np
from converge_load_forecasting import initialise,SDD_min_solar_mutiple_nodes,SDD_Same_Irrad_multiple_times,SDD_Same_Irrad_no_PV_houses_multiple_times,SDD_constant_PF_mutiple_nodes,SDD_known_pvs_multiple_nodes,SDD_using_temp_multilple_nodes,SDD_known_pvs_temp_multiple_node_algorithm

# # ==================================================
# # Initialize variables
# # ================================================== 
# # The first step is to create an input_features variable. It can have one of the two following formats.


# raw_data read from a server
NextGen_network_data_url = 'https://cloudstor.aarnet.edu.au/sender/download.php?token=087e5222-9919-4c67-af86-3e7d284e1ec2&files_ids=17805925'
raw_data = pd.read_csv(NextGen_network_data_url)
canberra_weather_url = 'https://cloudstor.aarnet.edu.au/sender/download.php?token=087e5222-9919-4c67-af86-3e7d284e1ec2&files_ids=17805920'
raw_weather_data = pd.read_csv(canberra_weather_url)
data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather, input_features = initialise(raw_data = raw_data,raw_weather_data=raw_weather_data)

# # Donwload if data is availbale in csv format
# customersdatapath = '/Users/mahdinoori/Documents/WorkFiles/Simulations/LoadForecasting/load_forecasting/data/Examples_data/NextGen_example.csv'
# weatherdatapath = '/Users/mahdinoori/Documents/WorkFiles/Simulations/LoadForecasting/load_forecasting/data/Examples_data/Canberra_weather_data.csv'
# data, customers_nmi,customers_nmi_with_pv,datetimes, customers, data_weather, input_features = initialise(customersdatapath = customersdatapath,weatherdatapath = weatherdatapath)

# some arbitarary parameters
n_customers = dict(take(20, customers.items()))     # take n customers from all the customer (to speed up the calculations)
time_step = 144
nmi = customers_nmi_with_pv[0] # an arbitary customer nmi
data_one_time = data.loc[pd.IndexSlice[:, datetimes[time_step]], :]   # data for a time step "time_step" 
Dates_for_plot_start = '2018-12-16'
Dates_for_plot_end = '2018-12-17'
time_steps_for_disagg = customers[customers_nmi[0]].data[Dates_for_plot_start:Dates_for_plot_end].index
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Examples of disaggregation functions
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================

# ################
# ## Technique 1: Minimum Solar Generation
# ################
res1 = SDD_min_solar_mutiple_nodes(n_customers,input_features) 
print('Disaggregation using Technique 1 is done!')
res1.to_csv('dissag_tech1.csv')

# ################
# ## Technique 2: Same Irradiance
# ################
res2 = SDD_Same_Irrad_multiple_times(data,input_features,time_steps_for_disagg,customers_nmi_with_pv)
print('Disaggregation using Technique 2 is done!')
res2.to_csv('dissag_tech2.csv')

# ################
# ## Technique 3: Same Irradiance and Houses Without PV Installation
# ################
customers_without_pv  = [customers_nmi_with_pv[i] for i in np.random.randint(0,len(customers_nmi_with_pv),10)]  # randomly select 10 nmi as nmi's without pv
customers_with_pv = [i for i in customers_nmi_with_pv if i not in customers_without_pv]
res3 = SDD_Same_Irrad_no_PV_houses_multiple_times(data,input_features,time_steps_for_disagg,customers_with_pv,customers_without_pv)
print('Disaggregation using Technique 3 is done!')
res3.to_csv('dissag_tech3.csv')

# ################
# ## Technique 4: Constant Power Factor Demand
# ################
res4 = SDD_constant_PF_mutiple_nodes(n_customers,input_features)
print('Disaggregation using Technique 4 is done!')
res4.to_csv('dissag_tech4.csv')

# ################
# ## Technique 5: Measurements from Neighbouring Sites
# ################
known_pv_nmis = [customers_nmi_with_pv[i] for i in np.random.randint(0,len(customers_nmi_with_pv),5)]
customers_known_pv = {i: customers[i] for i in known_pv_nmis}
res5 = SDD_known_pvs_multiple_nodes(customers,input_features,customers_known_pv,time_steps_for_disagg)
print('Disaggregation using Technique 5 is done!')
res5.to_csv('dissag_tech5.csv')


# ################
# ## Technique 6: Weather Data
# ################
res6 = SDD_using_temp_multilple_nodes(n_customers,input_features,data_weather)
print('Disaggregation using Technique 6 is done!')
res6.to_csv('dissag_tech6.csv')

# ################
# ## Technique 7: Proxy Measurements from Neighbouring Sites and Weather Data
# ################
known_pv_nmis = [customers_nmi_with_pv[i] for i in np.random.randint(0,len(customers_nmi_with_pv),5)]
customers_known_pv = {i: customers[i] for i in known_pv_nmis}
res7 = SDD_known_pvs_temp_multiple_node_algorithm(n_customers,input_features,data_weather,customers_known_pv,time_steps_for_disagg)
print('Disaggregation using Technique 7 is done!')
res7.to_csv('dissag_tech7.csv')