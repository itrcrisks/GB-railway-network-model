"""
Python code to create a train and passenger assignment model for Great Britain's railways network
- Using the train timetable information released by the ATOC: http://data.atoc.org/how-to
- Annual station usage statistics released by the ORR: https://dataportal.orr.gov.uk/statistics/usage/estimates-of-station-usage/
- A GIS representation of the GB railway network of nodes and edges
- File that match and map the ID columns between the ATOC, ORR and GIS network datasets  

The code outputs the daily passenger (total over 24 hours) OD matrix of flows for the rail network showing:
- origin_id: String node ID where journey starts
- destination_id: String node ID where journey ends
- node_path: List of node IDs traversed on the network between the OD pair
- node_path: List of edge IDs traversed on the network between the OD pair
- distance - Float distance in meters of the OD edge route (Based on shortest path algorithm)
- journeys - Float number of passenger jounreys made between the OD pair, aggregated over a day

The methodology is described in:
    Pant, R., Hall, J. W., & Blainey, S. P. (2016). 
    Vulnerability assessment framework for interdependent critical infrastructures: case-study for Great Britain’s rail network. 
    European Journal of Transport and Infrastructure Research, 16(1). 
    https://doi.org/10.18757/ejtir.2016.16.1.3120 

Author: Dr. Raghav Pant, Environmental Change Institute, University of Oxford

Compatibility: Python 3.6
"""
import sys
import os
import csv
import re
import ast
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import configparser
from tqdm import tqdm
tqdm.pandas()
#############################################################################################
# setup file locations and data files
# The project contents are organised in a base folder that contains the following subfolders:
# - timetable: The train timetable information as relased by ATOC
# - network: GIS node and edge files of the railway network
# - usage: The station usage statistics as released by the ORR
# - id_matches: CSV files that are created to match ATOC, Network, and ORR ID columns  
#############################################################################################

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']
TIMETABLE_PATH = os.path.join(BASE_PATH,'timetables')
NETWORK_PATH = os.path.join(BASE_PATH,'network')
USAGE_PATH = os.path.join(BASE_PATH,'usage')
ID_MATCH_PATH = os.path.join(BASE_PATH,'id_matches')
tqdm.pandas()

def extract_train_info(f_input):
    """Extract the train schedule information from the ATOC Timetable data as described in:
    http://data.atoc.org/sites/all/themes/atoc/files/RSPS5046.pdf

    Input parameters:
    -----------------
    f_input: String text with details of a train

    Result:
    -------
    train_dict: Dictionary containing details of: 
        - train_id - String unique train ID
        - date_from - Date from which the train runs in the year
        - date_to - Date to which the train runs in the year
        - cargo_type - Identify if it is an ordinary train or express train or bus or other
        - power_type - Identify if it is deisel or electric or HST or other
        - day (mon - sun) - Binary (0 or 1) to indicate if train runs on the day of the week
        - stp_indicator - Indicates if the train is cancelled, new, permanent or overlay  
    """
    name = f_input.split()
    transaction_type = name[0][2]
    train_id = name[0][3:9]
    Date_from = name[0][9:15]
    Date_to = name[0][15:21]
    Mon = int(name[0][21])
    Tue = int(name[0][22])
    Wed = int(name[0][23])
    Thu = int(name[0][24])
    Fri = int(name[0][25])
    Sat = int(name[0][26])
    Sun = int(name[0][27])

    if 'POL' in f_input or '1OL' in f_input or 'POU' in f_input or '1OU' in f_input or 'POO' in f_input or '1OO' in f_input or 'POS' in f_input or '1OS' in f_input or 'POW' in f_input or '1OW' in f_input:
        cargo_type = 'OrdP'
    elif 'PXC' in f_input or '1XC' in f_input or 'PXD' in f_input or '1XD' in f_input or 'PXI' in f_input or '1XI' in f_input or 'PXR' in f_input or '1XR' in f_input or 'PXU' in f_input or '1XU' in f_input or 'PXX' in f_input or '1XX' in f_input or 'PXZ' in f_input or '1XZ' in f_input:
        cargo_type = 'ExpP'
    elif 'BBR' in f_input or '5BR' in f_input or 'BBS' in f_input or '5BS' in f_input:
        cargo_type = 'BUS'
    else:
        cargo_type = 'Other'

    if 'DMU' in f_input:
        Power_type = 'DMU'
    elif 'EMU' in f_input:
        Power_type = 'EMU'
    elif 'HST' in f_input:
        Power_type = 'HST'
    else:
        Power_type = 'Other'

    STP_indicator = name[-1][0]

    train_dict = {
                    'transaction_type':transaction_type,
                    'train_id':train_id, 
                    'date_from':Date_from,
                    'date_to':Date_to,
                    'cargo_type': cargo_type, 
                    'mon':Mon, 
                    'tue':Tue,
                    'wed':Wed, 
                    'thu':Thu, 
                    'fri':Fri, 
                    'sat':Sat, 
                    'sun':Sun, 
                    'power_type':Power_type, 
                    'stp_indicator':STP_indicator
                }
    return train_dict

def path_estimation(f,s_pos, e_pos):
    """Extract the train Timetable path as described in:
    http://data.atoc.org/sites/all/themes/atoc/files/RSPS5046.pdf

    Input parameters:
    -----------------
    f: String multiple lines of text with details of all timepoint locations (TIPLOC) where train passes through
    s_pos: Integer position of the line from which we want to extract the string text data
    e_pos: Integer position of the line to which we want to extract the string text data 

    Result:
    -------
    Dictionary containing details of: 
        - path - List of string of all TIPLOC code IDs where train is recorded at
        - times - List of all times at the TIPLOCs where the train
        - stop - List to identfiy if the train stops (S) or not (N) at the TIPLOC
    """
    path = []
    time = []
    stop = []
    for lines in f[s_pos:e_pos]:
        if lines[0:2] in ['LO','LI','LT']:
            stext = lines[2:10]
            tiploc = stext.split()
            if tiploc[0][-1] in ['2','3'] and tiploc[0][0:-1] in path:
                path.append(tiploc[0][0:-1])
            else:
                path.append(tiploc[0])

            time.append((lines[10:14],lines[15:19]))
            if lines[10:15].strip() or lines[16:20].strip():
                stop.append('S')
            else:
                stop.append('N')

    return {'path':path,'times':time,'stops':stop}

def process_timetable_data(timetable_input,csv_file_write=False,timetable_output=''):
    """Extract the train schedule information from the ATOC Timetable data as described in:
    http://data.atoc.org/sites/all/themes/atoc/files/RSPS5046.pdf

    Input parameters:
    -----------------
    timetable_input: String path to ATOC .mca file
    csv_file_write: Boolean True or False condition to specify if output should be stored in a csv file. Default is False
    timetable_output: String path to output .csv file to store the extracted and prrocessed timetable data. Default is empty

    Result:
    -------
    ttis_df: Pandas Dataframe object containing train timetable information
    """
    f = list(open(timetable_input,'r'))
    loc = []
    crloc = []
    '''Find locations of the logs for train journeys''' 
    for x in range(len(f)):
        name = f[x].split()
        if name[0][:2] == 'BS' and name[-1][0] != 'C':
            loc.append(x)

    net_dict = []
    for x in range(len(loc)):
        train_dict = extract_train_info(f[loc[x]])

        s_pos = loc[x] + 1
        if x < len(loc) - 1:
            e_pos = loc[x + 1]
        else:
            e_pos = len(f)

        path_dict = path_estimation(f,s_pos, e_pos)

        net_dict.append({**train_dict,**path_dict})

        print ("Done with instance:", x)

    ttis_df = pd.DataFrame.from_dict(net_dict)
    if csv_file_write:
        ttis_df.to_csv(timetable_output,index=False)

    return ttis_df

def shorten_paths_to_stops(x):
    """Shorten the train TIPLOC path to only include locations where train stops
    
    Input parameters:
    -----------------
    x: Pandas dataframe object that should contain a column named path

    Result:
    -------
    path - List of TILOC IDs where train stops
    """
    path = []
    for p in range(len(x.path)):
        if x.stops[p] == 'S':
            path.append(x.path[p])

    return path

def extract_timetable_slice(timetable_df,cargo_type,station_tiplocs,
                        start_date=None,end_date=None,csv_file_write=False,timetable_output=''):
    """Extract a slice of the train timetable data between a starrt date and an end date 
    Combine all the unique train routes and add up the numbers of the trains running along each route

    Input parameters:
    -----------------
    timetable_df: Pandas dataframe containing the whole train timetable 
    cargo_type: String value to indicate the 'cargo_type' of the train
    station_tiplocs: List of TIPLOC IDs corresponding to only stations with passengers
    start_date: Integer format for a start date
    end_date: Integer format for an end date
    csv_file_write: Boolean True or False to indicate if the result should be written to a csv file
    timetable_output: String name of the csv file in the folder path where it should be stored

    Result:
    -------
    selected_path: Pandas Dataframe object containing the selected train timetable paths and their aggregated train numbers
    """
    if start_date == None and end_date == None:
        timetable_df = timetable_df[timetable_df['cargo_type'].isin(cargo_type)]
    else:
        if start_date == None:
            start_date = min([int(d) for d in timetable_df['date_from'].values.tolist()])
        elif end_date == None:
            end_date = max([int(d) for d in timetable_df['date_from'].values.tolist()])

        dates = [str(i) for i in range(start_date,end_date+1)]
        timetable_df = timetable_df[(timetable_df['cargo_type'].isin(cargo_type)) & (timetable_df['date_from'].isin(dates))]
    
    timetable_df['path_stops'] = timetable_df.progress_apply(lambda x: shorten_paths_to_stops(x),axis=1)
    timetable_df['path_stops'] = timetable_df.progress_apply(lambda x: str(x.path_stops),axis=1)
    timetable_df = timetable_df.groupby(['path_stops'])[['mon','tue','wed','thu','fri','sat','sun']].sum().reset_index()
    timetable_df['path_stops'] = timetable_df.progress_apply(lambda x:ast.literal_eval(x['path_stops']),axis=1)
    timetable_df['path_index'] = timetable_df.index.values.tolist()

    selected_paths = []
    for ttis in timetable_df.itertuples():
        train_stops = []
        for train_stop in ttis.path_stops:
            if train_stop in station_tiplocs:
                train_stops.append(train_stop)
        
        if train_stops:
            selected_paths.append((ttis.path_index,train_stops))
    
    selected_paths = pd.DataFrame(selected_paths,columns=['path_index','path_stops'])
    timetable_df.drop('path_stops',axis=1,inplace=True)
    selected_paths = pd.merge(selected_paths,timetable_df,how='left',on=['path_index'])
    selected_paths['path_stops'] = selected_paths.progress_apply(lambda x: str(x.path_stops),axis=1)
    selected_paths = selected_paths.groupby(['path_stops'])[['mon','tue','wed','thu','fri','sat','sun']].sum().reset_index()
    selected_paths['path_stops'] = selected_paths.progress_apply(lambda x:ast.literal_eval(x['path_stops']),axis=1)
    if csv_file_write == True:
        selected_paths.to_csv(timetable_output,index=False)

    return selected_paths

def get_train_stops_days(flow_dataframe,path_criteria):
    """Estimate the number of trains on each day of the week that start, end or make an intermediate stop at a TIPLOC

    Input parameters:
    -----------------
    flow_dataframe: Pandas dataframe with details of the weekly train timetable information 
    path_criteria: String column name that contains the TIPLOC path information of each train 

    Result:
    -------
    station_train_df: Pandas dataframe containing: 
        - tiploc - TIPLOC code IDs
        - origin_trains_day (mon-sun) - Numbers of trains orginiating from TIPLOC on a day of a week
        - destination_trains_day (mon-sun) - Numbers of trains terminating at TIPLOC on a day of a week
        - intermediate_trains_day (mon-sun) - Numbers of trains passing through TIPLOC on a day of a week
    """
    station_train_dict = []
    for v in flow_dataframe.itertuples():
        path = getattr(v,path_criteria)
        start = {
                    'tiploc': path[0],
                    'origin_trains_mon':v.mon,
                    'origin_trains_tue':v.tue,
                    'origin_trains_wed':v.wed,
                    'origin_trains_thu':v.thu,
                    'origin_trains_fri':v.fri,
                    'origin_trains_sat':v.sat,
                    'origin_trains_sun':v.sun
                }
        station_train_dict.append(start)
        end = {
                    'tiploc':path[-1],
                    'destination_trains_mon':v.mon,
                    'destination_trains_tue':v.tue,
                    'destination_trains_wed':v.wed,
                    'destination_trains_thu':v.thu,
                    'destination_trains_fri':v.fri,
                    'destination_trains_sat':v.sat,
                    'destination_trains_sun':v.sun
                }
        station_train_dict.append(end)
        for p in path[1:-1]:
            intermediate = {
                    'tiploc':p,
                    'intermediate_trains_mon':v.mon,
                    'intermediate_trains_tue':v.tue,
                    'intermediate_trains_wed':v.wed,
                    'intermediate_trains_thu':v.thu,
                    'intermediate_trains_fri':v.fri,
                    'intermediate_trains_sat':v.sat,
                    'intermediate_trains_sun':v.sun
                    }
            
            station_train_dict.append(intermediate)
    
    station_train_df = pd.DataFrame(station_train_dict).fillna(0)
    train_cols = [c for c in station_train_df.columns.values.tolist() if c != 'tiploc']
    
    return station_train_df.groupby(['tiploc'])[train_cols].sum().reset_index()

def create_networkx_from_dataframe(graph_dataframe, directed=False, simple=False):
    """Create a networkx graph object from a pandas dataframe

    Input parameters:
    -----------------
    graph_dataframe: Pandas dataframe with graph topology 
    directed: Boolean True or False for creating a directed graph object
    simple: Boolean True or False for creating a Graph, or DiGraph or MultiDiGraph object 

    Result:
    -------
    graph: Networkx graph object
    """
    if directed and simple:
        create_using = nx.DiGraph()
    elif directed and not simple:
        create_using = nx.MultiDiGraph()
    elif not directed and not simple:
        create_using = nx.MultiGraph()
    else:
        create_using = nx.Graph()

    graph = nx.from_pandas_edgelist(
        graph_dataframe,
        'from_node',
        'to_node',
        edge_attr=list(graph_dataframe.columns)[2:],
        create_using=create_using
    )

    es, vs, simple = graph.edges, graph.nodes, not graph.is_multigraph()
    d = "directed" if directed else "undirected"
    s = "simple" if simple else "multi"
    print(
        "Created {}, {} {}: {} edges, {} nodes.".format(
            s, d, "nxgraph", len(es), len(vs)))

    return graph

def path_attributes_networkx(graph, path, attribute):
    """Extract the attributes from a networkx graph object, given a node path 

    Input parameters:
    -----------------
    graph: Networkx graph object
    path: List of nodes that form a path on the graph
    attribute: Strring name of the attribute that should be in the networkx attrribute dictionary 

    Result:
    -------
    attr: List of the attribute values corresponding to the node path, extracted from the networkx object
    """
    attr = []
    ods = list(zip(path[:-1], path[1:]))
    for od in ods:
        attr += [
            d[attribute]
            for (u, v, d) in graph.edges(data = True)
            if (u, v) == od or (v, u) == od
        ]

    return attr

def path_to_route(path_dataframe,edge_dataframe,tiploc_mapping_nodes):
    """Map the TIPLOC route onto the network route by mapping TIPLOC to network nodes and
    Finding the equivalent route on a network for given node-node paths that might not be adjacent to each other
    By estimating the shortest path between consecutive nodes on the network

    Input parameters:
    -----------------
    path_dataframe: Pandas dataframe with details of the node-node path information
    edge_dataframe: Pandas dataframe with details of the network topology
    tiploc_mapping_nodes: Pandas dataframe with the details of the matches between TIPLOC IDs and node IDs 

    Result:
    -------
    path_dataframe: Pandas dataframe with details of the network routes 
                    with node IDs, edge IDs and distances along routes 
    """
    node_paths = []
    edge_paths = []
    distance_paths = []
    net = create_networkx_from_dataframe(edge_dataframe,directed=False)
    for path_data in path_dataframe.itertuples():
        node_path = []
        path = path_data.path_stops
        for p in range(len(path)-1):
            source = tiploc_mapping_nodes.loc[tiploc_mapping_nodes['tiploc']==path[p],'node_id'].values[0]
            target = tiploc_mapping_nodes.loc[tiploc_mapping_nodes['tiploc']==path[p+1],'node_id'].values[0]
            if source != target:
                pth = nx.shortest_path(net,source,target, weight = 'length')
            else:
                pth = [source]
            if len(node_path) == 0:
                for item in range(len(pth)):
                    node_path.append(pth[item])
            else:
                for item in range(len(pth)-1):
                    if pth[item+1] != node_path[-1]:
                        node_path.append(pth[item+1])
        
        edge_path = path_attributes_networkx(net,node_path,'edge_id')
        distance_path = [round(dist,3) for dist in path_attributes_networkx(net,node_path,'length')]
        
        node_paths.append(node_path)
        edge_paths.append(edge_path)
        distance_paths.append(distance_path)
        print ('Done with path {} out of {}'.format(path_data.Index,len(path_dataframe.index)))
    
    path_dataframe['node_path']=node_paths
    path_dataframe['edge_path']=edge_paths
    path_dataframe['distances']=distance_paths
    return path_dataframe

def main():
    ###################################################################
    # Specify all the input files and column names needed for the model
    ###################################################################
    timetable_data = os.path.join(TIMETABLE_PATH,'ttisf459.mca') # The .mca is text file as released by ATOC
    time_id_column = 'tiploc' # Name of the column assgined to station codes in the timetable data 
    rail_nodes = os.path.join(NETWORK_PATH,'rail_nodes.shp') # The rail node shapefile
    node_id_column = 'node_id' # Name of ID column in nodes shapefile
    rail_edges = os.path.join(NETWORK_PATH,'rail_edges.shp') # The rail edge shapefile
    # The rail edges file should have the following columns: ['from_node','to_node','edge_id']
    usage_data =  os.path.join(USAGE_PATH,'estimates-of-station-usage-2017-18.xlsx') # ORR releases data in xlsx format
    usage_sheet = 'Estimates of Station Usage' # Name of the excel sheet in the ORR data, which contains station annual usage data
    usage_id_column = 'TLC' # Name of ID column which should be in usage column. LAter renamed to tlc
    usage_entry = usage_exits = '1718 Entries & Exits' # Name of column(s) in ORR excel sheet contains statistics of annual entries and exits at stations
    usage_interchanges = '1718 Interchanges' # Name of column in ORR excel sheet contains statistics of annual interchanges at stations
    network_id_matches = os.path.join(ID_MATCH_PATH,'timetable_tiploc_crs_node_matching_final.csv') # To match network node IDs with timetable and station usage IDs
    
    ###################################################
    # Specify some input data parameters and conditions
    ###################################################
    cargo_type = ['OrdP','ExpP'] # Indicates that we only extract passenger train timetables from ATOC data 
    start_date = 190519 # We can set a start date to extract a weekly schedule of trains from the ATOC data
    end_date = 190525 # We can set an end date to extract a weekly schedule of trains from the ATOC data
    days = ['mon','tue','wed','thu','fri','sat','sun'] # Days in the week
    selected_day = 'wed' # Select a typical day of the working week for which we will estimate the OD matrix 

    ###############################################
    # Create output folder and specify output files
    ###############################################
    outputs = os.path.join(BASE_PATH, 'outputs')
    if os.path.exists(outputs) == False:
        os.mkdir(outputs)
    full_timetable = os.path.join(outputs,'train_ttis_info_2019.csv')
    store_full_timetable = False # Set to true if you want to store the output csv file
    combined_timetable = os.path.join(outputs,'train_ttis_paths_combined_2019.csv')
    store_combined_timetable = False # Set to true if you want to store the output csv file
    station_daily_usage = os.path.join(outputs,'station_daily_entry_exits.csv')
    store_station_daily_usage = False # Set to true if you want to store the output csv file
    rail_routes = os.path.join(outputs,'rail_paths.csv')
    store_rail_routes = False # Set to true if you want to store the output csv file
    od_matrix_output = os.path.join(outputs,'od_matrix.csv')
    store_od_matrix = True

    ######################################################################
    # Step 1: 
    # Assign the annual station usage statistics to the rail network nodes
    # Convert the station usage to weekly estimates
    ######################################################################
    id_file = pd.read_csv(network_id_matches) # Contains mapping between network IDs, timetable IDs and suage IDs
    print ('* Add station usage numbers of the network station nodes')
    nodes = gpd.read_file(rail_nodes)
    nodes = pd.merge(nodes[[node_id_column]],id_file,how='left',on=[node_id_column])

    station_usage = pd.read_excel(usage_data,sheet_name=usage_sheet,thousands=",",na_values=[":"," :"]).fillna(0)
    station_usage = station_usage[station_usage[usage_id_column] != 0]
    station_usage.rename(columns = {usage_id_column:'tlc'},inplace=True)
    usage_id_column = 'tlc'
    station_usage['entries'] = 0.5*station_usage[usage_entry]
    station_usage['exits'] = 0.5*station_usage[usage_exits]
    station_usage['interchanges'] = station_usage[usage_interchanges]
    nodes = pd.merge(nodes,
                    station_usage[[usage_id_column,'entries','exits','interchanges']],
                    on=[usage_id_column],how='left')
    nodes['weekly_entries'] = 1.0*(nodes['entries'] + nodes['interchanges'])/52.0
    nodes['weekly_entries'] = nodes['weekly_entries'].fillna(value=0)
    nodes['weekly_exits'] = 1.0*(nodes['exits'] + nodes['interchanges'])/52.0
    nodes['weekly_exits'] = nodes['weekly_exits'].fillna(value=0)
    del station_usage

    print ('* Process timetable data and extract the station stops')
    #########################################
    # Step 2: 
    # Extract the train timetable information
    #########################################
    timetable_df = process_timetable_data(timetable_data,
                                        csv_file_write=store_full_timetable,
                                        timetable_output=full_timetable)

    ##############################################################################################
    # Step 3: 
    # Extract the train timetable for the specific date range and passenger types
    # Truncate the timetable routes to only station stops 
    # Group all the unique train paths over a day and add up the numbers of trains along each path  
    ##############################################################################################
    station_stops = list(set(nodes[nodes['weekly_entries']>0][time_id_column].values.tolist()))
    # Decided not to set a start and end date because not all routes were being represented 
    timetable_df = extract_timetable_slice(timetable_df,cargo_type,station_stops,
                                        start_date=None,end_date=None,
                                        csv_file_write=store_combined_timetable,
                                        timetable_output=combined_timetable)
    
    ##############################################################################################
    # Step 4: 
    # Find the number of trains that start, end and make an intermeidate stop at each TIPLOC
    # Match this to the node IDs and the station usage numbers 
    # Find the total entries and exits along stations on a particular day   
    ##############################################################################################    
    print ('* Find station starts and intermediate and final stops')
    station_stops = get_train_stops_days(timetable_df,'path_stops')
    station_stops = pd.merge(station_stops,nodes[[time_id_column,
                                                node_id_column]],on=[time_id_column],how='left').fillna(0)
    station_stops = station_stops[station_stops[node_id_column] != 0]
    entry_exit_cols = [c for c in station_stops.columns.values.tolist() if c not in [node_id_column,time_id_column]]
    station_stops = station_stops.groupby([node_id_column])[entry_exit_cols].sum().reset_index()
    station_stops = pd.merge(station_stops,
                        nodes[[node_id_column,
                            'weekly_entries',
                            'weekly_exits']].drop_duplicates(subset=[node_id_column],keep='first'),
                            how='left',on=[node_id_column]).fillna(0)
    all_entries = []
    all_exits = []
    entry_cols = []
    exit_cols = []
    for day in days:
        entry_cols.append(['origin_trains_{}'.format(day),'intermediate_trains_{}'.format(day)])
        exit_cols.append(['destination_trains_{}'.format(day),'intermediate_trains_{}'.format(day)])
        all_entries += ['origin_trains_{}'.format(day),'intermediate_trains_{}'.format(day)]
        all_exits += ['destination_trains_{}'.format(day),'intermediate_trains_{}'.format(day)] 
    
    for d in range(len(days)):
        station_stops['entries_{}'.format(days[d])] = (station_stops[entry_cols[d]].sum(axis=1)/station_stops[all_entries].sum(axis=1))*station_stops['weekly_entries']
        station_stops['exits_{}'.format(days[d])] = (station_stops[exit_cols[d]].sum(axis=1)/station_stops[all_exits].sum(axis=1))*station_stops['weekly_exits']
    
    if store_station_daily_usage is True:
        station_stops.to_csv(station_daily_usage,index=False)

    node_flows = pd.merge(nodes[['node_id']].drop_duplicates(subset=['node_id'],keep='first'),
                                station_stops[['node_id','entries_{}'.format(selected_day),
                                'exits_{}'.format(selected_day)]],
                                how='left',on=['node_id']).fillna(0)
    node_flows.rename(columns={'entries_{}'.format(selected_day):'entries',
                            'exits_{}'.format(selected_day):'exits'},inplace=True)
    node_list = list(set(node_flows['node_id'].values.tolist()))
    
    ##########################################################################
    # Step 5: 
    # Convert the TIPLOC paths to actual network node, edge and distance paths
    ##########################################################################
    print ('* Find the network routes for the train timetables stops')
    edges = gpd.read_file(rail_edges)
    if 'length' in edges.columns.values.tolist():
        edges.drop('length',axis=1,inplace=True)
    
    edges['length'] = edges.progress_apply(lambda x: x.geometry.length,axis=1)
    paths_mapped = path_to_route(timetable_df, edges[['from_node','to_node','edge_id','length']],nodes)
    del timetable_df, edges, nodes
    
    paths_mapped['node_path'] = paths_mapped.progress_apply(lambda x: str(x.node_path),axis=1)
    paths_mapped = paths_mapped[paths_mapped.node_path != '[]'].reset_index()
    paths_mapped['path_index'] = paths_mapped.index.values.tolist()
    paths_mapped['node_path'] = paths_mapped.progress_apply(lambda x:ast.literal_eval(x.node_path),axis=1)

    ##########################################################################
    # Step 6: 
    # Find the attractiveness of each route for a particular starting stations
    # Which is the weight of the numbers of trains along different routes 
    # and the numbers of exits on the routes
    ##########################################################################
    
    print ('* Find the attractiveness of all routes')
    node_attract_all_routes = []
    node_attract_total = dict([(n,0) for n in node_list])
    for p in paths_mapped.itertuples():
        route = p.node_path
        if route: 
            node_attract_route = [0.0]*len(route)
            for item in range(len(route)-1):
                node_attract_route[item] = node_flows[node_flows['node_id'].isin(route[item+1:])]['exits'].sum()
                node_attract_route[item] = getattr(p,day)*float(node_attract_route[item])
                node_attract_total[route[item]] += float(node_attract_route[item])

            node_attract_all_routes.append(node_attract_route)
        
        else:
            node_attract_all_routes.append([])
        print ('Done with path:',p.Index)


    ##########################################################################
    # Step 7: 
    # Find the total entries from each station along each train route
    ##########################################################################
    print ('* Find the entries of all routes')
    node_entries_all_routes = []
    # route_entries_total = []
    for p in paths_mapped.itertuples():
        route = p.node_path
        if route:
            node_entry_route = [0.0]*len(route)
            for item in range(len(route)):
                if node_attract_total[route[item]] > 0:
                    node_entry_route[item] = float(1.0)*float(node_flows.loc[node_flows['node_id']==route[item],'entries'].values[0])*float(node_attract_all_routes[p.Index][item]/node_attract_total[route[item]])

            node_entries_all_routes.append(node_entry_route)
            # route_entries_total.append({'path_index':p.path_index,'od_flows':sum(node_entry_route)})
        else:
            node_entries_all_routes.append([])
            # route_entries_total.append({'path_index':p.path_index,'od_flows':0})
        
        print ('Done with path:',p.Index)
        
    
    # route_entries_total = pd.DataFrame(route_entries_total)
    # if 'od_flows' in paths_mapped.columns.values.tolist():
    #     paths_mapped.drop('od_flows',axis=1,inplace=True)
    # pd.merge(paths_mapped,route_entries_total,how='left',on=['path_index']).to_csv(os.path.join(outputs,'rail_paths.csv'),index=False)
    
    ################################################################
    # Step 7: 
    # Find the total exits from each station along each train route
    ################################################################
    print ('* Find the exits of all routes')
    node_exits_all_routes = []
    for p in paths_mapped.itertuples():
        route = p.node_path
        if route:
            node_exit_route = [0.0]*len(route)
            for item in range(len(route)):
                node_index = node_list.index(route[item])
                node_exit_route[item] = float(1.0)*float(node_flows.loc[node_flows['node_id']==route[item],'exits'].values[0])

            node_exits_all_routes.append(node_exit_route)

        print ('Done with path:',p.Index)

    paths_mapped['node_entries'] = node_entries_all_routes
    paths_mapped['node_exits'] = node_exits_all_routes

    if store_rail_routes is True: 
        paths_mapped.to_csv(rail_routes,index=False)
    
    ##########################################################################################
    # Step 8: 
    # Generate the node-node OD matrix with the routes, distances and daily passenger jounreys
    ##########################################################################################
    od_matrix = []
    for p in paths_mapped.itertuples():
        route = p.node_path
        if route:
            for st_pos in range(len(route)-1):
                exits = p.node_exits[st_pos+1:]
                if sum(exits) > 0:
                    start_end = list((1.0*p.node_entries[st_pos]*np.array(exits))/sum(exits))
                    end_st = route[st_pos+1:]
                    end_lines = p.edge_path[st_pos:]
                    start_st = route[st_pos]
                    end_dist = np.cumsum(p.distances[st_pos:])
                    od_vals = []
                    for k in range(len(end_st)):
                        od_vals.append((start_st,end_st[k],[start_st]+end_st[:k+1],end_lines[:k+1],end_dist[k],start_end[k]))

                    od_matrix += [k for k in od_vals if k[-1] > 0]

    od_matrix = pd.DataFrame(od_matrix,columns=['origin_id','destination_id','node_path','edge_path','distance','journeys'])
    od_matrix['node_path'] = od_matrix.progress_apply(lambda x:str(x.node_path),axis=1)
    od_matrix['edge_path'] = od_matrix.progress_apply(lambda x:str(x.edge_path),axis=1)
    od_matrix.groupby(['origin_id','destination_id',
        'node_path','edge_path','distance'])['journeys'].sum().reset_index()
    if store_od_matrix is True:
        od_matrix.to_csv(od_matrix_output,index=False)
    
if __name__ == "__main__":
    main()

