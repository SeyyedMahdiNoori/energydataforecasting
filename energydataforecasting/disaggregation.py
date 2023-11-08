
import pandas as pd
import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor
import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocess as mp
from pyomo.environ import NonNegativeReals, ConcreteModel, Var, Objective, Set, Constraint
from pyomo.opt import SolverFactory
import itertools
from dateutil.parser import parse
from dateutil.parser import ParserError

from typing import Union, Dict, Tuple, List

from converge_load_forecasting import Customers, prepare_proxy_data_for_training

# Warnings configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# A function to decide whether a string in the form of datetime has a time zone or not
def has_timezone_SDD(string: str) -> bool:
    '''
    has_timezone(string) accept string in the form of datetime and return True if it has timezone, and it returns False otherwise.
    '''
    try:
        if type(string) == str:
            parsed_date = parse(string)
            return parsed_date.tzinfo is not None
        elif type(string) == pd._libs.tslibs.timestamps.Timestamp:
            return string.tzinfo is not None
        else:
            return False
    except (TypeError, ValueError):
        return False

def pool_executor_parallel_SDD(function_name, repeat_iter, input_features):
    '''
    pool_executor_parallel_SDD(function_name,repeat_iter,input_features)
    
    This function is used to parallelised the forecasting for each nmi
    '''
    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(input_features)))  
    return results



# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================
# #                                                                                     Solar and Demand Disaggregation Algorithms
# # ==================================================================================================# # ==================================================================================================
# # ==================================================================================================# # ==================================================================================================


### The numbering for each technique refer to the numbering used in the associated article ("Customer-Level Solar-Demand Disaggregation: The Value of Information").
### Also, for more details on each approach, please refer to the above article. In what follows, we use SDD which stands for solar demand disaggregation


# # ================================================================
# # Technique 1: Minimum Solar Generation
# # ================================================================

def SDD_min_solar_single_node(customer,input_features):

    # print("Customer nmi: {nmi}, {precent}%".format(nmi = customer.nmi, precent = round((Customers.instances.index(customer.nmi) + 1) / len(Customers.instances) * 100, 1)))
    demand , solar = customer.Generate_disaggregation_positive_minimum_PV(input_features)

    result = pd.concat([demand, solar], axis=1, keys=['demand_disagg', 'pv_disagg'])
    result = result.rename_axis('datetime')
    result['nmi'] = [customer.nmi] * len(result)
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return(result)

def SDD_min_solar_mutiple_nodes(customers,input_features):

    predictions_prallel = pool_executor_parallel_SDD(SDD_min_solar_single_node,customers.values(),input_features)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return(predictions_prallel)

# # ================================================================
# # Technique 2: Same Irradiance
# # ================================================================

def SDD_Same_Irrad_single_time(time_step,customers,input_features):

    """
    SDD_Same_Irrad(t,customers_nmi_with_pv,datetimes,data_one_time), where t is the time-step of the disaggregation.
    
    This function disaggregates the demand and generation for all the nodes in the system at time-step t. 

    It is uses an optimisation algorithm with constrain:
        P_{t}^{pv} * PanleSize_{i} + P^{d}_{i,t}  == P^{agg}_{i,t} + P^{pen-p}_{i,t} - P^{pen-n}_{i,t},
    with the objective:
        min (P_{t}^{pv} + 10000 * \sum_{i} (P^{pen-p}_{i,t} - P^{pen-n}_{i,t}) 
    variables P^{pen-p}_{i,t} and P^{pen-n}_{i,t}) are defined to prevenet infeasibilities the optimisation problem, and are added to the objective function
    with a big coefficient. Variables P_{t}^{pv} and P^{d}_{i,t} denote the irradiance at time t, and demand at nmi i and time t, respectively. Also, parameters 
    PanleSize_{i} and P^{agg}_{i,t} denote the PV panel size of nmi i, and the recorded aggregated demand at nmi i and time t, respectively.
    """

    t = time_step
    
    customers_nmi_with_pv = list(customers.keys())
    datetimes = list(customers[customers_nmi_with_pv[0]].data.index)

    model=ConcreteModel()
    model.Time = Set(initialize=range(t,t+1))
    model.pv=Var(model.Time, bounds=(0,1))
    model.demand=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_p=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_n=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t] * customers[i].data.pv_system_size[0] == customers[i].data[input_features["Forecasted_param"]][datetimes[t]] + model.penalty_p[t,i] - model.penalty_n[t,i] 
    model.cons = Constraint(model.Time,customers_nmi_with_pv,rule=load_balance)

    # # Objective
    def obj_rule(model):
        return sum(model.pv[t] for t in model.Time) + 10000 * sum( sum( model.penalty_p[t,i] + model.penalty_n[t,i] for i in customers_nmi_with_pv ) for t in model.Time)
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    print(" Disaggregating {first}-th time step".format(first = t))

    result_output_temp =  ({i:    (model.pv[t].value * customers[i].data.pv_system_size[0] + model.penalty_p[t,i].value)  for i in customers_nmi_with_pv},
            {i:      model.demand[t,i].value + model.penalty_n[t,i].value  for i in customers_nmi_with_pv} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)

    return result_output

def pool_executor_parallel_time(function_name,repeat_iter,customers,input_features):

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(customers),itertools.repeat(input_features)))  
    return results

def SDD_Same_Irrad_multiple_times(datetimes,customers,input_features):

    """
    Generate_disaggregation_optimisation()
    
    This function disaggregates the demand and generation for all the nodes in the system and all the time-steps, and adds the disaggergations to each
    class variable. It applies the disaggregation to all nmis. This fuction uses function "pool_executor_disaggregation" to run the disaggregation algorithm.  
    """

    predictions_prallel = pool_executor_parallel_time(SDD_Same_Irrad_single_time,range(0,len(datetimes)),customers,input_features)
    
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    print('Done')

    return predictions_prallel

# # ================================================================
# # Technique 3: Same Irradiance and Houses Without PV Installation
# # ================================================================
def SDD_Same_Irrad_no_PV_houses_single_time(time_step,customers,customers_with_pv_nmi,customers_without_pv_nmi,input_features):
    
    t = time_step
    print(" Disaggregating {first}-th time step".format(first = t))

    datetimes = list(customers[list(customers.keys())[0]].data.index)

    model=ConcreteModel()

    model.Time = Set(initialize=range(t,t+1))
    model.pv = Var(model.Time,customers_with_pv_nmi, bounds=(0,1))
    model.absLoad = Var(model.Time, within=NonNegativeReals)
    model.demand = Var(model.Time,customers_with_pv_nmi,within=NonNegativeReals)
    model.penalty_p = Var(model.Time,customers_with_pv_nmi,within=NonNegativeReals)
    model.penalty_n = Var(model.Time,customers_with_pv_nmi,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t,i] * customers[i].data.pv_system_size[0] == customers[i].data[input_features["Forecasted_param"]][t]
    model.cons = Constraint(model.Time,customers_with_pv_nmi,rule=load_balance)

    def abs_Load_1(model,t,i):
        return model.absLoad[t] >= sum(model.demand[t,i] for i in customers_with_pv_nmi)/len(customers_with_pv_nmi) - sum(customers[i].data.load_active[datetimes[t]] for i in customers_without_pv_nmi )/len(customers_without_pv_nmi)
    model.cons_abs1 = Constraint(model.Time,customers_with_pv_nmi,rule=abs_Load_1)

    def abs_Load_2(model,t,i):
        return model.absLoad[t] >=  sum(customers[i].data.load_active[datetimes[t]] for i in customers_without_pv_nmi )/len(customers_without_pv_nmi) - sum(model.demand[t,i] for i in customers_with_pv_nmi)/len(customers_with_pv_nmi)
    model.cons_abs2 = Constraint(model.Time,customers_with_pv_nmi,rule=abs_Load_2)

    # # Objective
    def obj_rule(model):
        return (  model.absLoad[t] + sum(model.pv[t,i]**2 for i in customers_with_pv_nmi)/len(customers_with_pv_nmi) )
    # def obj_rule(model):
    #     return (  sum(model.demand[t,i] for i in customers_with_pv_nmi)/len(customers_with_pv_nmi) - sum(data_one_time.loc[i].load_active[datetimes[t]]/len(customers_without_pv_nmi) for i in customers_without_pv_nmi) 
    #             + sum(model.pv[t,i]**2 for i in customers_with_pv_nmi) 
    #             )
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    result_output_temp =  ({i:    (model.pv[t,i].value * customers[i].data.pv_system_size[0])  for i in customers_with_pv_nmi},
            {i:      model.demand[t,i].value  for i in customers_with_pv_nmi} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)

    return result_output



def pool_executor_parallel_time_no_PV_houses(function_name,repeat_iter,customers,customers_with_pv_nmi,customers_without_pv_nmi,input_features):
    

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(customers),itertools.repeat(customers_with_pv_nmi),itertools.repeat(customers_without_pv_nmi),itertools.repeat(input_features)))  
    return results


def SDD_Same_Irrad_no_PV_houses_multiple_times(datetimes,customers,customers_with_pv_nmi,customers_without_pv_nmi,input_features):

    """
    Generate_disaggregation_optimisation()
    
    This function disaggregates the demand and generation for all the nodes in the system and all the time-steps, and adds the disaggergations to each
    class variable. It applies the disaggregation to all nmis. This fuction uses function "pool_executor_disaggregation" to run the disaggregation algorithm.  
    """

    predictions_prallel = pool_executor_parallel_time_no_PV_houses(SDD_Same_Irrad_no_PV_houses_single_time,range(0,len(datetimes)),customers,customers_with_pv_nmi,customers_without_pv_nmi,input_features)
    
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    print('Done')

    return predictions_prallel

# # ================================================================
# # Technique 4: Constant Power Factor Demand
# # ================================================================

def SDD_constant_PF_single_node(customer,input_features):

    print("Customer nmi: {nmi}, {precent}%".format(nmi = customer.nmi, precent = round((Customers.instances.index(customer.nmi) + 1) / len(Customers.instances) * 100, 1)))

    demand , solar = customer.generate_disaggregation_using_reactive(input_features)

    result = pd.concat([demand, solar], axis=1, keys=['demand_disagg', 'pv_disagg'])
    result = result.rename_axis('datetime')
    result['nmi'] = [customer.nmi] * len(result)
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)

    return(result)

def SDD_constant_PF_mutiple_nodes(customers,input_features):

    predictions_prallel = pool_executor_parallel_SDD(SDD_constant_PF_single_node,customers.values(),input_features)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return(predictions_prallel)


# # ================================================================
# # Technique 5: Measurements from Neighbouring Sites
# # ================================================================
def SDD_known_pvs_single_node(customer,customers_known_pv,datetimes):

    print("Customer nmi: {nmi}, {precent}%".format(nmi = customer.nmi, precent = round((Customers.instances.index(customer.nmi) + 1) / len(Customers.instances) * 100, 1)))

    model=ConcreteModel()
    known_pv_nmis = list(customers_known_pv.keys())
    model.pv_cites = Set(initialize=known_pv_nmis)
    model.Time = Set(initialize=range(0,len(datetimes)))
    model.weight = Var(model.pv_cites, bounds=(0,1))

    # # Constraints
    def load_balance(model):
        return sum(model.weight[i] for i in model.pv_cites) == 1 
    model.cons = Constraint(rule=load_balance)

    # Objective
    def obj_rule(model):
        return  sum(
        ( sum(model.weight[i] * customers_known_pv[i].data.pv[datetimes[t]]/customers_known_pv[i].data.pv_system_size[0] for i in model.pv_cites)
                - max(-customer.data.active_power[datetimes[t]],0)/customer.data.pv_system_size[0]
        )**2 for t in model.Time)

    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)
     
    pv_dis = pd.concat([sum(model.weight[i].value * customers_known_pv[i].data.pv/customers_known_pv[i].data.pv_system_size[0] for i in model.pv_cites) * customer.data.pv_system_size[0],
                    -customer.data.active_power]).max(level=0)
    
    load_dis = customer.data.active_power + pv_dis

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    result = result.rename_axis('datetime')
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)

def pool_executor_parallel_knownPVS(function_name,repeat_iter,customers_known_pv,datetimes,input_features):

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(customers_known_pv),itertools.repeat(datetimes)))  
    return results


def SDD_known_pvs_multiple_nodes(customers,input_features,customers_known_pv,datetimes):


    predictions_prallel = pool_executor_parallel_knownPVS(SDD_known_pvs_single_node,customers.values(),customers_known_pv,datetimes,input_features)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    return(predictions_prallel)


# # ================================================================
# # Technique 6: Weather Data
# # ================================================================
def SDD_using_temp_single_node(customer,datetimes,data_weather,input_features):

    print("Customer nmi: {nmi}, {precent}%".format(nmi = customer.nmi, precent = round((Customers.instances.index(customer.nmi) + 1) / len(Customers.instances) * 100, 1)))

    weather_input = prepare_proxy_data_for_training(customer.data.loc[datetimes].index,data_weather)
    pv_dis = - customer.data[input_features["Forecasted_param"]].loc[datetimes].clip(upper = 0)
    load_dis = customer.data.active_power[datetimes] + pv_dis

    iteration = 0
    pv_dis_iter = copy.deepcopy(pv_dis*0)
    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 15:

        iteration += 1
        pv_dis_iter = copy.deepcopy(pv_dis)

        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=customer.data.active_power[datetimes].index)
        load_dis[load_dis < 0 ] = 0 
        pv_dis = load_dis - customer.data.active_power[datetimes]

    pv_dis[pv_dis < 0 ] = 0 
    load_dis =  customer.data.active_power[datetimes] + pv_dis

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    result = result.rename_axis('datetime')
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)

def pool_executor_parallel_temperature(function_name,repeat_iter,datetimes,data_weather,input_features):
    
    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(datetimes),itertools.repeat(data_weather),itertools.repeat(input_features)))  
    return results

def SDD_using_temp_multilple_nodes(customers,datetimes,data_weather,input_features):

    predictions_prallel = pool_executor_parallel_temperature(SDD_using_temp_single_node,customers.values(),datetimes,data_weather,input_features)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)
    
    return predictions_prallel


# # ================================================================
# # Technique 7: Proxy Measurements from Neighbouring Sites and Weather Data
# # ================================================================
def SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,pv_iter):
    known_pv_nmis = list(customers_known_pv.keys())
    model=ConcreteModel()
    model.pv_cites = Set(initialize=known_pv_nmis)
    model.Time = Set(initialize=range(0,len(datetimes)))
    model.weight=Var(model.pv_cites, bounds=(0,1))

    # # Constraints
    def load_balance(model):
        return sum(model.weight[i] for i in model.pv_cites) == 1 
    model.cons = Constraint(rule=load_balance)

    # Objective
    def obj_rule(model):
        return  sum(
                    (sum(model.weight[i] * customers_known_pv[i].data.pv[datetimes[t]]/customers_known_pv[i].data.pv_system_size[0] for i in model.pv_cites)
                        - pv_iter[datetimes[t]]/customer.data.pv_system_size[0] )**2 for t in model.Time)

    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)
    
    return pd.concat([sum(model.weight[i].value * customers_known_pv[i].data.pv[datetimes]/customers_known_pv[i].data.pv_system_size[0] for i in model.pv_cites) * customer.data.pv_system_size[0],
                    -customer.data.active_power[datetimes]]).max(level=0)

def SDD_known_pvs_temp_single_node_algorithm(customer,datetimes,data_weather,input_features,customers_known_pv):
    
    print("Customer nmi: {nmi}, {precent}%".format(nmi = customer.nmi, precent = round((Customers.instances.index(customer.nmi) + 1) / len(Customers.instances) * 100, 1)))

    weather_input = prepare_proxy_data_for_training(customer.data.loc[datetimes].index,data_weather)    
    
    pv_iter0 = - customer.data[input_features["Forecasted_param"]].loc[datetimes].clip(upper = 0)

    pv_dis = SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,pv_iter0)
    load_dis = customer.data.active_power[datetimes] + pv_dis
    
    iteration = 0
    pv_dis_iter = copy.deepcopy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 15:

        iteration += 1
        pv_dis_iter = copy.deepcopy(pv_dis)
        
        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=pv_dis.index)
        pv_dis = load_dis - customer.data.active_power[datetimes]
        pv_dis[pv_dis < 0 ] = 0 
        pv_dis = SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,pv_dis)
        load_dis = customer.data.active_power[datetimes] + pv_dis

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    result = result.rename_axis('datetime')
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)


def pool_executor_parallel_temperature_known_pv(function_name,repeat_iter,datetimes,data_weather,input_features,customers_known_pv):
    
    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(datetimes),itertools.repeat(data_weather),itertools.repeat(input_features),itertools.repeat(customers_known_pv)))  
    return results

def SDD_known_pvs_temp_multiple_node_algorithm(customers,datetimes,data_weather,input_features,customers_known_pv):

    predictions_prallel = pool_executor_parallel_temperature_known_pv(SDD_known_pvs_temp_single_node_algorithm,customers.values(),datetimes,data_weather,input_features,customers_known_pv)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)
    
    return predictions_prallel
