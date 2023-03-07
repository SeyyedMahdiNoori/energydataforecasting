
import pandas as pd
import numpy as np
import copy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import  DecisionTreeRegressor
import skforecast 
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocess as mp
from pyomo.environ import NonNegativeReals, ConcreteModel, Var, Objective, Set, Constraint
from pyomo.opt import SolverFactory
import tqdm
from functools import partialmethod
import itertools
import connectorx as cx
import tsprial
import dateutil
from dateutil.parser import parse
from dateutil.parser import ParserError

from typing import Union, Dict, Tuple, List


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

    print(f'customer_ID: {customer.nmi} begin')

    customer.Generate_disaggregation_positive_minimum_PV()

    result = pd.DataFrame(customer.data.pv_disagg)
    result['demand_disagg'] = customer.data.demand_disagg
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
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

def SDD_Same_Irrad_single_time(time_step,customers_nmi_with_pv,datetimes,data_one_time):

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

    model=ConcreteModel()
    model.Time = Set(initialize=range(t,t+1))
    model.pv=Var(model.Time, bounds=(0,1))
    model.demand=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_p=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_n=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]] + model.penalty_p[t,i] - model.penalty_n[t,i] 
    model.cons = Constraint(model.Time,customers_nmi_with_pv,rule=load_balance)

    # # Objective
    def obj_rule(model):
        return sum(model.pv[t] for t in model.Time) + 10000 * sum( sum( model.penalty_p[t,i] + model.penalty_n[t,i] for i in customers_nmi_with_pv ) for t in model.Time)
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    print(" Disaggregating {first}-th time step".format(first = t))
    # print(t)

    result_output_temp =  ({i:    (model.pv[t].value * data_one_time.loc[i].pv_system_size[0] + model.penalty_p[t,i].value)  for i in customers_nmi_with_pv},
            {i:      model.demand[t,i].value + model.penalty_n[t,i].value  for i in customers_nmi_with_pv} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)
    
    # result_output = pd.concat({datetimes[t]: result_output}, names=['datetime'])

    return result_output

def SDD_Same_Irrad_for_parallel(time_step,customers_nmi_with_pv,datetimes):

    """
    disaggregate_demand(t,customers_nmi_with_pv,customers), where t is the time-step of the disaggregation.
    
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
    data_one_time = shared_data_disaggregation_optimisation.loc[pd.IndexSlice[:, datetimes[t]], :]

    model=ConcreteModel()
    model.Time = Set(initialize=range(t,t+1))
    model.pv=Var(model.Time, bounds=(0,1))
    model.demand=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_p=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)
    model.penalty_n=Var(model.Time,customers_nmi_with_pv,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]] + model.penalty_p[t,i] - model.penalty_n[t,i] 
    model.cons = Constraint(model.Time,customers_nmi_with_pv,rule=load_balance)

    # # Objective
    def obj_rule(model):
        return sum(model.pv[t] for t in model.Time) + 10000 * sum( sum( model.penalty_p[t,i] + model.penalty_n[t,i] for i in customers_nmi_with_pv ) for t in model.Time)
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    print(" Disaggregating {first}-th time step".format(first = t))

    result_output_temp =  ({i:    (model.pv[t].value * data_one_time.loc[i].pv_system_size[0] + model.penalty_p[t,i].value)  for i in customers_nmi_with_pv},
            {i:      model.demand[t,i].value + model.penalty_n[t,i].value  for i in customers_nmi_with_pv} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)

    return result_output

def pool_executor_parallel_time(function_name,repeat_iter,customers_nmi_with_pv,datetimes,data,input_features):
    
    global shared_data_disaggregation_optimisation

    shared_data_disaggregation_optimisation = copy.deepcopy(data)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(customers_nmi_with_pv),itertools.repeat(datetimes)))  
    return results

def SDD_Same_Irrad_multiple_times(data,input_features,datetimes,customers_nmi_with_pv):

    """
    Generate_disaggregation_optimisation()
    
    This function disaggregates the demand and generation for all the nodes in the system and all the time-steps, and adds the disaggergations to each
    class variable. It applies the disaggregation to all nmis. This fuction uses function "pool_executor_disaggregation" to run the disaggregation algorithm.  
    """

    global shared_data_disaggregation_optimisation

    predictions_prallel = pool_executor_parallel_time(SDD_Same_Irrad_for_parallel,range(0,len(datetimes)),customers_nmi_with_pv,datetimes,data,input_features)
    
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    print('Done')

    # print(len(predictions_prallel))
    
    if 'shared_data_disaggregation_optimisation' in globals():
        del(shared_data_disaggregation_optimisation)

    return predictions_prallel

# # ================================================================
# # Technique 3: Same Irradiance and Houses Without PV Installation
# # ================================================================
def SDD_Same_Irrad_no_PV_houses_single_time(time_step,data,customers_with_pv,customers_without_pv,datetimes):
    
    t = time_step

    model=ConcreteModel()

    data_one_time = data.loc[pd.IndexSlice[:, datetimes[t]], :]

    model.Time = Set(initialize=range(t,t+1))
    model.pv = Var(model.Time,customers_with_pv, bounds=(0,1))
    model.absLoad = Var(model.Time, within=NonNegativeReals)
    model.demand = Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_p = Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_n = Var(model.Time,customers_with_pv,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t,i] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]]
    model.cons = Constraint(model.Time,customers_with_pv,rule=load_balance)

    def abs_Load_1(model,t,i):
        return model.absLoad[t] >= sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]] for i in customers_without_pv )/len(customers_without_pv)
    model.cons_abs1 = Constraint(model.Time,customers_with_pv,rule=abs_Load_1)

    def abs_Load_2(model,t,i):
        return model.absLoad[t] >=  sum(data_one_time.loc[i].load_active[datetimes[t]] for i in customers_without_pv )/len(customers_without_pv) - sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv)
    model.cons_abs2 = Constraint(model.Time,customers_with_pv,rule=abs_Load_2)

    # # Objective
    def obj_rule(model):
        return (  model.absLoad[t] + sum(model.pv[t,i]**2 for i in customers_with_pv)/len(customers_with_pv) )
    # def obj_rule(model):
    #     return (  sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]]/len(customers_without_pv) for i in customers_without_pv) 
    #             + sum(model.pv[t,i]**2 for i in customers_with_pv) 
    #             )
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    result_output_temp =  ({i:    (model.pv[t,i].value * data_one_time.loc[i].pv_system_size[0])  for i in customers_with_pv},
            {i:      model.demand[t,i].value  for i in customers_with_pv} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)

    return result_output



def pool_executor_parallel_time_no_PV_houses(function_name,repeat_iter,customers_with_pv,customers_without_pv,datetimes,data,input_features):
    
    global shared_data_disaggregation_optimisation_no_PV

    shared_data_disaggregation_optimisation_no_PV = copy.deepcopy(data)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(customers_with_pv),itertools.repeat(customers_without_pv),itertools.repeat(datetimes)))  
    return results


def SDD_Same_Irrad_no_PV_houses_multiple_times(data,input_features,datetimes,customers_with_pv,customers_without_pv):

    """
    Generate_disaggregation_optimisation()
    
    This function disaggregates the demand and generation for all the nodes in the system and all the time-steps, and adds the disaggergations to each
    class variable. It applies the disaggregation to all nmis. This fuction uses function "pool_executor_disaggregation" to run the disaggregation algorithm.  
    """

    global shared_data_disaggregation_optimisation_no_PV

    predictions_prallel = pool_executor_parallel_time_no_PV_houses(SDD_Same_Irrad_no_PV_houses_single_time_for_parallel,range(0,len(datetimes)),customers_with_pv,customers_without_pv,datetimes,data,input_features)
    
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    print('Done')

    # print(len(predictions_prallel))
    
    if 'shared_data_disaggregation_optimisation_no_PV' in globals():
        del(shared_data_disaggregation_optimisation_no_PV)

    return predictions_prallel


def SDD_Same_Irrad_no_PV_houses_single_time_for_parallel(time_step,customers_with_pv,customers_without_pv,datetimes):

    t = time_step

    model=ConcreteModel()

    data_one_time = shared_data_disaggregation_optimisation_no_PV.loc[pd.IndexSlice[:, datetimes[t]], :]

    model.Time = Set(initialize=range(t,t+1))
    model.pv = Var(model.Time,customers_with_pv, bounds=(0,1))
    model.absLoad = Var(model.Time, within=NonNegativeReals)
    model.demand = Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_p = Var(model.Time,customers_with_pv,within=NonNegativeReals)
    model.penalty_n = Var(model.Time,customers_with_pv,within=NonNegativeReals)

    # # Constraints
    def load_balance(model,t,i):
        return model.demand[t,i] - model.pv[t,i] * data_one_time.loc[i].pv_system_size[0] == data_one_time.loc[i].active_power[datetimes[t]]
    model.cons = Constraint(model.Time,customers_with_pv,rule=load_balance)

    def abs_Load_1(model,t,i):
        return model.absLoad[t] >= sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]] for i in customers_without_pv )/len(customers_without_pv)
    model.cons_abs1 = Constraint(model.Time,customers_with_pv,rule=abs_Load_1)

    def abs_Load_2(model,t,i):
        return model.absLoad[t] >=  sum(data_one_time.loc[i].load_active[datetimes[t]] for i in customers_without_pv )/len(customers_without_pv) - sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv)
    model.cons_abs2 = Constraint(model.Time,customers_with_pv,rule=abs_Load_2)

    # # Objective
    def obj_rule(model):
        return (  model.absLoad[t] + sum(model.pv[t,i]**2 for i in customers_with_pv)/len(customers_with_pv) )
    # def obj_rule(model):
    #     return (  sum(model.demand[t,i] for i in customers_with_pv)/len(customers_with_pv) - sum(data_one_time.loc[i].load_active[datetimes[t]]/len(customers_without_pv) for i in customers_without_pv) 
    #             + sum(model.pv[t,i]**2 for i in customers_with_pv) 
    #             )
    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)

    result_output_temp =  ({i:    (model.pv[t,i].value * data_one_time.loc[i].pv_system_size[0])  for i in customers_with_pv},
            {i:      model.demand[t,i].value  for i in customers_with_pv} )

    result_output = pd.DataFrame.from_dict(result_output_temp[0], orient='index').rename(columns={0: 'pv_disagg'})
    result_output['demand_disagg'] = result_output_temp[1].values()    
    result_output.index.names = ['nmi']
    datetime = [datetimes[t]] * len(result_output)
    result_output['datetime'] = datetime
    result_output.reset_index(inplace=True)
    result_output.set_index(['nmi', 'datetime'], inplace=True)

    print(" Disaggregating {first}-th time step".format(first = t))

    return result_output


# # ================================================================
# # Technique 4: Constant Power Factor Demand
# # ================================================================

def SDD_constant_PF_single_node(customer,input_features):

    customer.generate_disaggregation_using_reactive(input_features)

    result = pd.DataFrame(customer.data.pv_disagg)
    result['demand_disagg'] = customer.data.demand_disagg
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
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
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)

def SDD_known_pvs_single_node_for_parallel(customer,datetimes):

    print(f'customer_ID: {customer.nmi} begin')

    model=ConcreteModel()
    known_pv_nmis = list(customers_known_pv_shared.keys())
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
        ( sum(model.weight[i] * customers_known_pv_shared[i].data.pv[datetimes[t]]/customers_known_pv_shared[i].data.pv_system_size[0] for i in model.pv_cites)
                - max(-customer.data.active_power[datetimes[t]],0)/customer.data.pv_system_size[0]
        )**2 for t in model.Time)

    model.obj=Objective(rule=obj_rule)

    # # Solve the model
    opt = SolverFactory('gurobi')
    opt.solve(model)
     
    pv_dis = pd.concat([sum(model.weight[i].value * customers_known_pv_shared[i].data.pv/customers_known_pv_shared[i].data.pv_system_size[0] for i in model.pv_cites) * customer.data.pv_system_size[0],
                    -customer.data.active_power]).max(level=0)
    
    load_dis = customer.data.active_power + pv_dis

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)


def pool_executor_parallel_knownPVS(function_name,repeat_iter,input_features,customers_known_pv,datetimes):
    
    global customers_known_pv_shared

    customers_known_pv_shared = copy.deepcopy(customers_known_pv)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(datetimes)))  
    return results


def SDD_known_pvs_multiple_nodes(customers,input_features,customers_known_pv,datetimes):


    global customers_known_pv_shared

    predictions_prallel = pool_executor_parallel_knownPVS(SDD_known_pvs_single_node_for_parallel,customers.values(),input_features,customers_known_pv,datetimes)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    if 'customers_known_pv_shared' in globals():
        del(customers_known_pv_shared)

    return(predictions_prallel)


# # ================================================================
# # Technique 6: Weather Data
# # ================================================================
def SDD_using_temp_single_node(customer,datetimes,weatherdatapath=None,raw_weather_data=None):


    # read and process weather data if it has been inputted
    if weatherdatapath is None and raw_weather_data is None:
        data_weather = pd.DataFrame()
        return("Error!!! weather data is not provided")
    elif weatherdatapath is not None:
        data_weather = pd.read_csv(weatherdatapath)
    else:
        data_weather = copy.deepcopy(raw_weather_data)
        
    data_weather.rename(columns={"PeriodStart": "datetime"},inplace=True)
    data_weather = data_weather.drop('PeriodEnd', axis=1)

    # # ###### Pre-process the data ######
    # format datetime to pandas datetime format
    try:
        check_time_zone = has_timezone_SDD(data_weather.datetime[0])
    except AttributeError:
        print('Error!!! Input data is not the correct format! It should have a column with "datetime", a column with name "nmi" and at least one more column which is going to be forecasted')
        return pd.DataFrame() # To match the number of outputs

    try:
        if check_time_zone == False:
            data_weather['datetime'] = pd.to_datetime(data_weather['datetime'])
        else:
            data_weather['datetime'] = pd.to_datetime(data_weather['datetime'], utc=True, infer_datetime_format=True).dt.tz_convert("Australia/Sydney")
    except ParserError:
        print('Error!!! data.datetime should be a string that can be meaningfully changed to time.')
        return pd.DataFrame() # To match the number of outputs

    data_weather.set_index('datetime', inplace=True)
    
    data_weather['minute'] = data_weather.index.minute
    data_weather['hour'] = data_weather.index.hour
    data_weather['isweekend'] = (data_weather.index.day_of_week > 4).astype(int)
    data_weather['Temp_EWMA'] = data_weather.AirTemp.ewm(com=0.5).mean()        
    
    # data_weather.set_index(data_weather.index.tz_localize(None),inplace=True)
    data_weather = data_weather[~data_weather.index.duplicated(keep='first')]

    if has_timezone_SDD(customer.data.index[0]) == False and check_time_zone == True:
        data_weather.index = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x in data_weather.index.strftime("%Y-%m-%d %H:%M:%S")]

    # remove rows that have a different index from datetimes (main data index). This keeps them with the same lenght later on when the 
    # weather data is going to be used for learning
    set_diff = list( set(data_weather.index)-set( datetimes) )
    data_weather = data_weather.drop(set_diff)
    
    # fill empty rows (rows that are in the main data and not available in the weather data) with average over the same day.
    set_diff = list( set( datetimes) - set(data_weather.index) )
    for i in range(0,len(set_diff)):
        try:
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().AirTemp,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().Temp_EWMA,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                ],ignore_index=False)
        except Exception:
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':17.5,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':17.5,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                    ],ignore_index=False)
    
    weather_input = data_weather[['AirTemp','hour','minute','Temp_EWMA','isweekend']]
    
    pv_dis = copy.deepcopy(customer.data.active_power[datetimes])
    pv_dis[pv_dis > 0 ] = 0 
    pv_dis = -pv_dis
    load_dis = customer.data.active_power[datetimes] + pv_dis

    iteration = 0
    pv_dis_iter = copy.deepcopy(pv_dis*0)
    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 15:

        iteration += 1
        pv_dis_iter = copy.deepcopy(pv_dis)
        print(f'Iteration: {iteration}')

        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=customer.data.active_power[datetimes].index)
        load_dis[load_dis < 0 ] = 0 
        pv_dis = load_dis - customer.data.active_power[datetimes]

    pv_dis[pv_dis < 0 ] = 0 
    load_dis =  customer.data.active_power[datetimes] + pv_dis

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)

def SDD_using_temp_single_node_for_parallel(customer,datetimes):

    print(f'customer_ID: {customer.nmi}')

    weather_input = shared_weather_data[['AirTemp','hour','minute','Temp_EWMA','isweekend']]

    pv_dis = copy.deepcopy(customer.data.active_power[datetimes])
    pv_dis[pv_dis > 0 ] = 0 
    pv_dis = -pv_dis
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
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)

def pool_executor_parallel_temperature(function_name,repeat_iter,input_features,data_weather,datetimes):
    

    global shared_weather_data

    shared_weather_data = copy.deepcopy(data_weather)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(datetimes)))  
    return results

def SDD_using_temp_multilple_nodes(customers,input_features,datetimes,weatherdatapath=None,raw_weather_data=None):

    # read and process weather data if it has been inputted
    if weatherdatapath is None and raw_weather_data is None:
        data_weather = pd.DataFrame()
        return("Error!!! weather data is not provided")
    elif weatherdatapath is not None:
        data_weather = pd.read_csv(weatherdatapath)
    else:
        data_weather = copy.deepcopy(raw_weather_data)
        
    data_weather.rename(columns={"PeriodStart": "datetime"},inplace=True)
    data_weather = data_weather.drop('PeriodEnd', axis=1)

    # # ###### Pre-process the data ######
    # format datetime to pandas datetime format
    try:
        check_time_zone = has_timezone_SDD(data_weather.datetime[0])
    except AttributeError:
        print('Error!!! Input data is not the correct format! It should have a column with "datetime", a column with name "nmi" and at least one more column which is going to be forecasted')
        return pd.DataFrame() # To match the number of outputs

    try:
        if check_time_zone == False:
            data_weather['datetime'] = pd.to_datetime(data_weather['datetime'])
        else:
            data_weather['datetime'] = pd.to_datetime(data_weather['datetime'], utc=True, infer_datetime_format=True).dt.tz_convert("Australia/Sydney")
    except ParserError:
        print('Error!!! data.datetime should be a string that can be meaningfully changed to time.')
        return pd.DataFrame() # To match the number of outputs

    data_weather.set_index('datetime', inplace=True)
    
    data_weather['minute'] = data_weather.index.minute
    data_weather['hour'] = data_weather.index.hour
    data_weather['isweekend'] = (data_weather.index.day_of_week > 4).astype(int)
    data_weather['Temp_EWMA'] = data_weather.AirTemp.ewm(com=0.5).mean()        
    
    # data_weather.set_index(data_weather.index.tz_localize(None),inplace=True)
    data_weather = data_weather[~data_weather.index.duplicated(keep='first')]

    if has_timezone_SDD(customers[list(customers.keys())[0]].data.index[0]) == False and check_time_zone == True:
        data_weather.index = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x in data_weather.index.strftime("%Y-%m-%d %H:%M:%S")]

    # remove rows that have a different index from datetimes (main data index). This keeps them with the same lenght later on when the 
    # weather data is going to be used for learning
    set_diff = list( set(data_weather.index)-set( datetimes) )
    data_weather = data_weather.drop(set_diff)
    
    # fill empty rows (rows that are in the main data and not available in the weather data) with average over the same day.
    set_diff = list( set( datetimes) - set(data_weather.index) )
    for i in range(0,len(set_diff)):
        try:
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().AirTemp,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().Temp_EWMA,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                ],ignore_index=False)
        except Exception:
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':17.5,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':17.5,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                    ],ignore_index=False)
                                    
    global shared_weather_data

    predictions_prallel = pool_executor_parallel_temperature(SDD_using_temp_single_node_for_parallel,customers.values(),input_features,data_weather,datetimes)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    if 'shared_weather_data' in globals():
        del(shared_weather_data)

    return(predictions_prallel)


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

def SDD_known_pvs_temp_single_node_algorithm(customer,customers_known_pv,datetimes,weatherdatapath=None,raw_weather_data=None):
    
    # read and process weather data if it has been inputted
    if weatherdatapath is None and raw_weather_data is None:
        data_weather = pd.DataFrame()
        return("Error!!! weather data is not provided")
    elif weatherdatapath is not None:
        data_weather = pd.read_csv(weatherdatapath)
    else:
        data_weather = copy.deepcopy(raw_weather_data)
        
    data_weather.rename(columns={"PeriodStart": "datetime"},inplace=True)
    data_weather = data_weather.drop('PeriodEnd', axis=1)

    # # ###### Pre-process the data ######
    # format datetime to pandas datetime format
    try:
        check_time_zone = has_timezone_SDD(data_weather.datetime[0])
    except AttributeError:
        print('Error!!! Input data is not the correct format! It should have a column with "datetime", a column with name "nmi" and at least one more column which is going to be forecasted')
        return pd.DataFrame() # To match the number of outputs

    try:
        if check_time_zone == False:
            data_weather['datetime'] = pd.to_datetime(data_weather['datetime'])
        else:
            data_weather['datetime'] = pd.to_datetime(data_weather['datetime'], utc=True, infer_datetime_format=True).dt.tz_convert("Australia/Sydney")
    except ParserError:
        print('Error!!! data.datetime should be a string that can be meaningfully changed to time.')
        return pd.DataFrame() # To match the number of outputs

    data_weather.set_index('datetime', inplace=True)
    
    data_weather['minute'] = data_weather.index.minute
    data_weather['hour'] = data_weather.index.hour
    data_weather['isweekend'] = (data_weather.index.day_of_week > 4).astype(int)
    data_weather['Temp_EWMA'] = data_weather.AirTemp.ewm(com=0.5).mean()        
    
    # data_weather.set_index(data_weather.index.tz_localize(None),inplace=True)
    data_weather = data_weather[~data_weather.index.duplicated(keep='first')]

    if has_timezone_SDD(customer.data.index[0]) == False and check_time_zone == True:
        data_weather.index = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x in data_weather.index.strftime("%Y-%m-%d %H:%M:%S")]

    # remove rows that have a different index from datetimes (main data index). This keeps them with the same lenght later on when the 
    # weather data is going to be used for learning
    set_diff = list( set(data_weather.index)-set( datetimes) )
    data_weather = data_weather.drop(set_diff)
    
    # fill empty rows (rows that are in the main data and not available in the weather data) with average over the same day.
    set_diff = list( set( datetimes) - set(data_weather.index) )
    for i in range(0,len(set_diff)):
        try:
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().AirTemp,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().Temp_EWMA,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                ],ignore_index=False)
        except Exception:
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':17.5,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':17.5,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                    ],ignore_index=False)


    weather_input = data_weather[['AirTemp','hour','minute','Temp_EWMA','isweekend']]
    
    pv_iter0 = copy.deepcopy(customer.data.active_power[datetimes])
    pv_iter0[pv_iter0 > 0 ] = 0 
    pv_iter0 = -pv_iter0

    pv_dis = SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,pv_iter0)
    load_dis = customer.data.active_power[datetimes] + pv_dis
    
    iteration = 0
    pv_dis_iter = copy.deepcopy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 15:

        iteration += 1
        pv_dis_iter = copy.deepcopy(pv_dis)
        print(f'Iteration: {iteration}')
        
        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=pv_dis.index)
        pv_dis = load_dis - customer.data.active_power[datetimes]
        pv_dis[pv_dis < 0 ] = 0 
        pv_dis = SDD_known_pvs_temp_single_node(customer,customers_known_pv,datetimes,pv_dis)
        load_dis = customer.data.active_power[datetimes] + pv_dis

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)


def SDD_known_pvs_temp_single_node_algorithm_for_parallel(customer,datetimes):
    
    weather_input = shared_weather_data[['AirTemp','hour','minute','Temp_EWMA','isweekend']]

    pv_iter0 = copy.deepcopy(customer.data.active_power)
    pv_iter0[pv_iter0 > 0 ] = 0 
    pv_iter0 = -pv_iter0

    pv_dis = SDD_known_pvs_temp_single_node(customer,shared_data_known_pv,datetimes,pv_iter0)
    
    print(f'customer_ID: {customer.nmi} begin')
    load_dis = customer.data.active_power[datetimes] + pv_dis


    iteration = 0
    pv_dis_iter = copy.deepcopy(pv_dis*0)

    while (pv_dis_iter-pv_dis).abs().max() > 0.01 and iteration < 15:
        
        iteration += 1
        pv_dis_iter = copy.deepcopy(pv_dis)
        # print(iteration)
        
        regr = RandomForestRegressor(max_depth=24*12, random_state=0)
        regr.fit(weather_input.values, load_dis.values)
        load_dis = pd.Series(regr.predict(weather_input.values),index=pv_dis.index)
        pv_dis = SDD_known_pvs_temp_single_node(customer,shared_data_known_pv,datetimes,load_dis - customer.data.active_power[datetimes])
        load_dis = customer.data.active_power[datetimes] + pv_dis

    print(f'customer_ID: {customer.nmi} done!')

    result =  pd.DataFrame(data={'pv_disagg': pv_dis,'demand_disagg': load_dis})
    nmi = [customer.nmi] * len(result)
    result['nmi'] = nmi
    result.reset_index(inplace=True)
    result.set_index(['nmi', 'datetime'], inplace=True)
    return (result)


def pool_executor_parallel_known_pvs_temp(function_name,repeat_iter,input_features,data_weather,customers_known_pv,datetimes):
    
    global shared_data_known_pv
    global shared_weather_data

    shared_data_known_pv = copy.deepcopy(customers_known_pv)
    shared_weather_data = copy.deepcopy(data_weather)

    with ProcessPoolExecutor(max_workers=input_features['core_usage'],mp_context=mp.get_context('fork')) as executor:
        results = list(executor.map(function_name,repeat_iter,itertools.repeat(datetimes)))  
    return results


def SDD_known_pvs_temp_multiple_node_algorithm(customers,input_features,customers_known_pv,datetimes,weatherdatapath=None,raw_weather_data=None):

    # read and process weather data if it has been inputted
    if weatherdatapath is None and raw_weather_data is None:
        data_weather = pd.DataFrame()
        return("Error!!! weather data is not provided")
    elif weatherdatapath is not None:
        data_weather = pd.read_csv(weatherdatapath)
    else:
        data_weather = copy.deepcopy(raw_weather_data)
        
    data_weather.rename(columns={"PeriodStart": "datetime"},inplace=True)
    data_weather = data_weather.drop('PeriodEnd', axis=1)

    # # ###### Pre-process the data ######
    # format datetime to pandas datetime format
    try:
        check_time_zone = has_timezone_SDD(data_weather.datetime[0])
    except AttributeError:
        print('Error!!! Input data is not the correct format! It should have a column with "datetime", a column with name "nmi" and at least one more column which is going to be forecasted')
        return pd.DataFrame() # To match the number of outputs

    try:
        if check_time_zone == False:
            data_weather['datetime'] = pd.to_datetime(data_weather['datetime'])
        else:
            data_weather['datetime'] = pd.to_datetime(data_weather['datetime'], utc=True, infer_datetime_format=True).dt.tz_convert("Australia/Sydney")
    except ParserError:
        print('Error!!! data.datetime should be a string that can be meaningfully changed to time.')
        return pd.DataFrame() # To match the number of outputs

    data_weather.set_index('datetime', inplace=True)
    
    data_weather['minute'] = data_weather.index.minute
    data_weather['hour'] = data_weather.index.hour
    data_weather['isweekend'] = (data_weather.index.day_of_week > 4).astype(int)
    data_weather['Temp_EWMA'] = data_weather.AirTemp.ewm(com=0.5).mean()        
    
    # data_weather.set_index(data_weather.index.tz_localize(None),inplace=True)
    data_weather = data_weather[~data_weather.index.duplicated(keep='first')]

    if has_timezone_SDD(customers[list(customers.keys())[0]].data.index[0]) == False and check_time_zone == True:
        data_weather.index = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x in data_weather.index.strftime("%Y-%m-%d %H:%M:%S")]

    # remove rows that have a different index from datetimes (main data index). This keeps them with the same lenght later on when the 
    # weather data is going to be used for learning
    set_diff = list( set(data_weather.index)-set( datetimes) )
    data_weather = data_weather.drop(set_diff)
    
    # fill empty rows (rows that are in the main data and not available in the weather data) with average over the same day.
    set_diff = list( set( datetimes) - set(data_weather.index) )
    for i in range(0,len(set_diff)):
        try:
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().AirTemp,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':data_weather[set_diff[i].date().strftime('%Y-%m-%d')].mean().Temp_EWMA,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                ],ignore_index=False)
        except Exception:
            data_weather = pd.concat([data_weather,pd.DataFrame({'AirTemp':17.5,'hour':set_diff[i].hour,'minute':set_diff[i].minute,'Temp_EWMA':17.5,'isweekend':int((set_diff[i].day_of_week > 4))},index=[set_diff[i]])
                                    ],ignore_index=False)


    global shared_data_known_pv
    global shared_weather_data

    predictions_prallel = pool_executor_parallel_known_pvs_temp(SDD_known_pvs_temp_single_node_algorithm_for_parallel,customers.values(),input_features,data_weather,customers_known_pv,datetimes)
    predictions_prallel = pd.concat(predictions_prallel, axis=0)

    if 'shared_data_known_pv' in globals():
        del(shared_data_known_pv)
    if 'shared_weather_data' in globals():
        del(shared_weather_data)

    return(predictions_prallel)
