"""
Python code to find the failure impact of a cluster failure on the railway network
Author: Raghav Pant, ECI, University of Oxford
Date: February 05, 2016
"""
import sys
import os
import ast
import pandas as pd
import geopandas as gpd
import igraph as ig
import networkx as nx
from collections import defaultdict
import numpy as np
import configparser
from tqdm import tqdm
tqdm.pandas()

#############################################################################################
# setup file locations and data files
# The project contents are organised in a base folder that contains the following subfolders:
# - network: Folder containing GIS node and edge files of the railway network
# - outputs: The OD matrix result folder 
#############################################################################################

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']
NETWORK_PATH = os.path.join(BASE_PATH,'network')
FLOW_PATH = os.path.join(BASE_PATH,'outputs')
tqdm.pandas()
CRS = 'epsg:27700'

def get_flow_paths_indexes_of_elements(flow_dataframe,path_criteria):
	"""Estimate flow path indexes of all elements

    Parameters
    ---------
    flow_dataframe - Pandas DataFrame of list of node/edge paths
    path_criteria - String name of column of node/edge paths in flow dataframe

    Returns
    -------
    path_index - Dictionary of asset ID's as keys and list of flow path indexes as values
    """
    tqdm.pandas()
    path_index = defaultdict(list)
    for k,v in zip(chain.from_iterable(flow_dataframe[path_criteria].ravel()), flow_dataframe.index.repeat(flow_dataframe[path_criteria].str.len()).tolist()):
        path_index[k].append(v)

    return path_index

def get_flow_paths_indexes_of_failed_elements(asset_failure_set,
    flow_path_indexes):
    """Estimate flow path indexes of failed elements

    Parameters
    ---------
    asset_failure_set - List of string asset ID's
    flow_path_indexes - Dictionary of asset ID's as keys and list of flow path indexes as values 

    Returns
    -------
    List of all path indexes affected by the elements of the asset failure set
    """

    return list(set(list(chain.from_iterable([path_idx for path_key,path_idx in flow_path_indexes.items() if path_key in asset_failure_set]))))

def modify_igraph_with_removed_assets(network_df_in, asset_failure_set,
    asset_type='edge'):
    """Estimate network impacts of each failures
    When the tariff costs of each path are fixed by vehicle weight

    Parameters
    ---------
    network_df_in - Pandas DataFrame of network
    asset_failure_set - List of string edge ID's
    flow_dataframe - Pandas DataFrame of list of edge paths
    path_criteria - String name of column of edge paths in flow dataframe
    cost_criteria - String name of column of path costs in flow dataframe

    Returns
    -------
    edge_failure_dictionary : list[dict]
        With attributes
        edge_id - String name or list of failed edges
        origin - String node ID of Origin of disrupted OD flow
        destination - String node ID of Destination of disrupted OD flow
        no_access - Boolean 1 (no reroutng) or 0 (rerouting)
        new_cost - Float value of estimated cost of OD journey after disruption
        new_distance - Float value of estimated distance of OD journey after disruption
        new_path - List of string edge ID's of estimated new route of OD journey after disruption
        new_time - Float value of estimated time of OD journey after disruption
    """
    # edge_fail_dictionary = []

    # edge_path_index = list(set(list(chain.from_iterable([path_idx for path_key,path_idx in edge_flow_path_indexes.items() if path_key in asset_failure_set]))))

    if asset_type == 'node':
    	return ig.Graph.TupleList(network_df_in[~(network_df_in['from_node'].isin(asset_failure_set) | network_df_in['to_node'].isin(asset_failure_set))].itertuples(
            index=False), edge_attrs=list(network_df_in.columns)[2:])
    else:
        return ig.Graph.TupleList(network_df_in[~network_df_in['edge_id'].isin(asset_failure_set)].itertuples(
            index=False), edge_attrs=list(network_df_in.columns)[2:])

def igraph_scenario_edge_failures(network_df_in, asset_failure_set,
    flow_dataframe,edge_flow_path_indexes, path_criteria,
    cost_criteria,new_path = True):
    """Estimate network impacts of each failures
    When the tariff costs of each path are fixed by vehicle weight

    Parameters
    ---------
    network_df_in - Pandas DataFrame of network
    asset_failure_set - List of string edge ID's
    flow_dataframe - Pandas DataFrame of list of edge paths
    path_criteria - String name of column of edge paths in flow dataframe
    cost_criteria - String name of column of path costs in flow dataframe

    Returns
    -------
    edge_failure_dictionary : list[dict]
        With attributes
        edge_id - String name or list of failed edges
        origin - String node ID of Origin of disrupted OD flow
        destination - String node ID of Destination of disrupted OD flow
        no_access - Boolean 1 (no reroutng) or 0 (rerouting)
        new_cost - Float value of estimated cost of OD journey after disruption
        new_distance - Float value of estimated distance of OD journey after disruption
        new_path - List of string edge ID's of estimated new route of OD journey after disruption
        new_time - Float value of estimated time of OD journey after disruption
    """
    # edge_fail_dictionary = []

    # edge_path_index = list(set(list(chain.from_iterable([path_idx for path_key,path_idx in edge_flow_path_indexes.items() if path_key in asset_failure_set]))))

    if edge_path_index:
        select_flows = flow_dataframe[flow_dataframe.index.isin(edge_path_index)]
        del edge_path_index
        network_graph = ig.Graph.TupleList(network_df_in[~network_df_in['edge_id'].isin(asset_failure_set)].itertuples(
            index=False), edge_attrs=list(network_df_in.columns)[2:])

        first_edge_id = asset_failure_set[0]
        del asset_failure_set
        A = sorted(network_graph.clusters().subgraphs(),key=lambda l:len(l.es['edge_id']),reverse=True)
        access_flows = []
        edge_fail_dictionary = []
        for i in range(len(A)):
            network_graph = A[i]
            nodes_name = np.asarray([x['name'] for x in network_graph.vs])
            po_access = select_flows[(select_flows['origin_id'].isin(nodes_name)) & (
                    select_flows['destination_id'].isin(nodes_name))]

            if len(po_access.index) > 0:
                po_access = po_access.set_index('origin_id')
                origins = list(set(po_access.index.values.tolist()))
                for o in range(len(origins)):
                    origin = origins[o]
                    destinations = po_access.loc[[origin], 'destination_id'].values.tolist()
                    tons = po_access.loc[[origin], tons_criteria].values.tolist()
                    paths = network_graph.get_shortest_paths(
                        origin, destinations, weights=cost_criteria, output="epath")
                    if new_path == True:
                        for p in range(len(paths)):
                            new_dist = 0
                            new_time = 0
                            new_gcost = 0
                            new_path = []
                            for n in paths[p]:
                                new_dist += network_graph.es[n]['length']
                                new_time += network_graph.es[n][time_criteria]
                                new_gcost += network_graph.es[n][cost_criteria]
                                new_path.append(network_graph.es[n]['edge_id'])
                            edge_fail_dictionary.append({'edge_id': first_edge_id, 'origin_id': origin, 'destination_id': destinations[p],
                                                         'new_path':new_path,'new_distance': new_dist, 'new_time': new_time,
                                                         'new_cost': tons[p]*new_gcost, 'no_access': 0})
                    else:
                        for p in range(len(paths)):
                            new_dist = 0
                            new_time = 0
                            new_gcost = 0
                            for n in paths[p]:
                                new_dist += network_graph.es[n]['length']
                                new_time += network_graph.es[n][time_criteria]
                                new_gcost += network_graph.es[n][cost_criteria]
                            edge_fail_dictionary.append({'edge_id': first_edge_id, 'origin_id': origin, 'destination_id': destinations[p],
                                                         'new_path':[],'new_distance': new_dist, 'new_time': new_time,
                                                         'new_cost': tons[p]*new_gcost, 'no_access': 0})
                    del destinations, tons, paths
                del origins
                po_access = po_access.reset_index()
                po_access['access'] = 1
                access_flows.append(po_access[['origin_id','destination_id','access']])
            del po_access

        del A

        if len(access_flows):
            access_flows = pd.concat(access_flows,axis=0,sort='False', ignore_index=True)
            select_flows = pd.merge(select_flows,access_flows,how='left',on=['origin_id','destination_id']).fillna(0)
        else:
            select_flows['access'] = 0

        no_access = select_flows[select_flows['access'] == 0]
        if len(no_access.index) > 0:
            for value in no_access.itertuples():
                edge_fail_dictionary.append({'edge_id': first_edge_id, 'origin_id': getattr(value,'origin_id'),
                                            'destination_id': getattr(value,'destination_id'),
                                            'new_path':[],'new_distance': 0, 'new_time': 0, 'new_cost': 0, 'no_access': 1})

        del no_access, select_flows

    return edge_fail_dictionary

def main():
	######################
    # Read the input files
    ######################
    rail_nodes = os.path.join(FLOW_PATH,'rail_nodes.shp') # The rail node shapefile
    node_id_column = 'node_id' # Name of ID column in nodes shapefile
    rail_edges = os.path.join(FLOW_PATH,'rail_edges_flows.shp') # The rail edge shapefile
    edge_id_column = 'edge_id' # Name of ID column in edges shapefile

    od_matrix_output = os.path.join(FLOW_PATH,'od_matrix.csv') # The rail OD matrix result csv file
    node_path_column = 'node_path'  # Name of the column containing the node paths
    edge_path_column = 'edge_path' # Name of the column containing the edge paths
    flow_column = 'journeys' # Name of the column containing the daily trips

    ###############################################
    # Create output folder and specify output files
    ###############################################
    outputs = os.path.join(BASE_PATH, 'outputs')
    if os.path.exists(outputs) == False:
        os.mkdir(outputs)
    edge_failure_output = os.path.join(outputs,'rail_edges_failures.csv')
    store_edge_failure_output = True # Set to true if you want to store the output csv file
    node_failure_output = os.path.join(outputs,'rail_nodes_failures.shp')
    store_node_failure_output = True # Set to true if you want to store the output csv file

    ################################################################################
    # Get the OD matrix paths and find all paths corresponding to each node and edge 
    ################################################################################
    flow_paths = pd.read_csv(od_matrix_output)
    flow_paths[node_path_column] = flow_paths.progress_apply(lambda x:ast.literal_eval(x[node_path_column]),axis=1)
    flow_paths[edge_path_column] = flow_paths.progress_apply(lambda x:ast.literal_eval(x[edge_path_column]),axis=1)
    node_indexes = get_path_indexes(flow_paths,node_path_column)
    edge_indexes = get_path_indexes(flow_paths,edge_path_column)

    ######################
    # Get the rail network
    ######################
    edges = gpd.read_file(rail_edges)
    # Eliminate all edges with 0 flows, because they do not have any trains through them
    # This is an artifact of the network data ih which some rail lines exist, which arre no longer in use  
    edges = edges[edges[flow_column]>0]






if __name__ == "__main__":

    main()