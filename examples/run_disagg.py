# # ==================================================================================================
# # This package is developed as a part of the Converge project at the Australian National University
# # Users can use this package to generate forecasting values for electricity demand. It can also be 
# # used to disaggregate solar and demand from the aggregated measurement at the connection point
# # to the system
# # For further details on the disaggregation algorithms, please refer to the user manual file
# #
# # Below is an example of the main functionalities of the solar demand disaggregation component of the package
# # ==================================================================================================

import numpy as np
from converge_load_forecasting import (
    SDD_constant_PF_mutiple_nodes, SDD_known_pvs_multiple_nodes,
    SDD_known_pvs_temp_multiple_node_algorithm, SDD_min_solar_mutiple_nodes,
    SDD_Same_Irrad_multiple_times, SDD_Same_Irrad_no_PV_houses_multiple_times,
    SDD_using_temp_multilple_nodes, initialise)

# # ==================================================
# # Initialize variables
# # ================================================== 
# # The first step is to create an input_features variable. It can have one of the two following formats.

# # Donwload if data is availbale in csv format
customersdatapath = './NextGen_example.csv'
weatherdatapath = './Canberra_weather_data.csv'
data, customers, input_features, customers_nmi, datetimes = initialise(customersdatapath = customersdatapath,core_usage=8)

import copy
customers_nmi_with_pv = copy.deepcopy(customers_nmi)

# some arbitarary parameters
# time_steps_for_disagg = copy(datetimes)
Dates_for_plot_start = '2018-12-23'
Dates_for_plot_end = '2018-12-23'
time_steps_for_disagg = customers[customers_nmi[0]].data[Dates_for_plot_start:Dates_for_plot_end].index

customers_without_pv  = [customers_nmi_with_pv[i] for i in np.random.default_rng(seed=1).choice(len(customers_nmi_with_pv), size=40, replace=False)]  # randomly select 10 nmi as nmi's without pv
customers_with_pv = [i for i in customers_nmi_with_pv if i not in customers_without_pv]
known_pv_nmis = [customers_with_pv[i] for i in np.random.default_rng(seed=10).choice(len(customers_with_pv), size=3, replace=False)]
customers_known_pv = {i: customers[i] for i in known_pv_nmis}
n_customers = {i: customers[i] for i in customers_with_pv if i not in customers_known_pv}

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
res2 = SDD_Same_Irrad_multiple_times(data,input_features,time_steps_for_disagg,list(n_customers.keys()))
print('Disaggregation using Technique 2 is done!')
res2.to_csv('dissag_tech2.csv')

# ################
# ## Technique 3: Same Irradiance and Houses Without PV Installation
# ################
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
res5 = SDD_known_pvs_multiple_nodes(customers,input_features,customers_known_pv,time_steps_for_disagg)
print('Disaggregation using Technique 5 is done!')
res5.to_csv('dissag_tech5.csv')


# ################
# ## Technique 6: Weather Data
# ################
res6 = SDD_using_temp_multilple_nodes(n_customers,input_features,time_steps_for_disagg,weatherdatapath=weatherdatapath)
print('Disaggregation using Technique 6 is done!')
res6.to_csv('dissag_tech6.csv')

# # ################
# # ## Technique 7: Proxy Measurements from Neighbouring Sites and Weather Data
# # ################
res7 = SDD_known_pvs_temp_multiple_node_algorithm(n_customers,input_features,customers_known_pv,time_steps_for_disagg,weatherdatapath = weatherdatapath)
print('Disaggregation using Technique 7 is done!')
res7.to_csv('dissag_tech7.csv')