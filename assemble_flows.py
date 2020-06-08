"""Assemble network flow paths with weighted flows on the rail node and edge network of GB
    Takes the OD matrix result generated by the rail_od_flows.py
"""
import sys
import os
import ast
import pandas as pd
import geopandas as gpd
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
#####################################
# READ MAIN DATA
#####################################
def flow_paths_to_network_assets(paths_df,path_id_column,path_flow_id_column,gdf_file,gdf_file_id_column):
    """Add OD flow paths value of nodes/edges

    Outputs ``gdf_edges`` - a shapefile with od flows of all
    each node/edge of network.

    Parameters
    ---------
    paths_df
        Pandas DataFrame of OD flow paths and their flow
    path_id_column
        String name of th enode/edgeee path id
    gdf_edges
        GeoDataFrame of network edge set
    save_csv
        Boolean condition to tell code to save created edge csv file
    save_shapes
        Boolean condition to tell code to save created edge shapefile
    shape_output_path
        Path where the output shapefile will be stored
    csv_output_path
        Path where the output csv file will be stored

    """
    path_flow = defaultdict(float)
    for row in paths_df.itertuples():
        for item in getattr(row,path_id_column):
            path_flow[item] += getattr(row,path_flow_id_column)

    path_flow = pd.DataFrame(path_flow.items(),columns=[gdf_file_id_column,path_flow_id_column])
    gdf_file = pd.merge(gdf_file,path_flow,how='left',on=[gdf_file_id_column])
    gdf_file[path_flow_id_column].fillna(0,inplace=True)

    return gpd.GeoDataFrame(gdf_file,geometry='geometry',crs=CRS)
    
    
def main():
    rail_nodes = os.path.join(NETWORK_PATH,'rail_nodes.shp') # The rail node shapefile
    node_id_column = 'node_id' # Name of ID column in nodes shapefile
    rail_edges = os.path.join(NETWORK_PATH,'rail_edges.shp') # The rail edge shapefile
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
    edge_flows_output = os.path.join(outputs,'rail_edges_flows.shp')
    store_edge_flows_output = True # Set to true if you want to store the output csv file
    node_flows_output = os.path.join(outputs,'rail_nodes_flows.shp')
    store_node_flows_output = True # Set to true if you want to store the output csv file

    ######################################
    # Get the OD matrix paths and journeys 
    ######################################
    flow_paths = pd.read_csv(od_matrix_output)
    flow_paths[node_path_column] = flow_paths.progress_apply(lambda x:ast.literal_eval(x[node_path_column]),axis=1)
    flow_paths[edge_path_column] = flow_paths.progress_apply(lambda x:ast.literal_eval(x[edge_path_column]),axis=1)

    ###########################################
    # Find total jounreys along nodes and edges 
    ###########################################
    edges = gpd.read_file(rail_edges)
    edges = flow_paths_to_network_assets(flow_paths,
                                    edge_path_column,
                                    flow_column,
                                    edges,
                                    edge_id_column)      

    if store_edge_flows_output is True:
        edges.to_file(edge_flows_output)
    
    nodes = gpd.read_file(rail_nodes)
    nodes = flow_paths_to_network_assets(flow_paths,
                                    node_path_column,
                                    flow_column,
                                    nodes,
                                    node_id_column)      

    if store_node_flows_output is True:
        nodes.to_file(node_flows_output)
if __name__ == "__main__":

    main()