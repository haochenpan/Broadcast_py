from enum import Enum
import pickle
import networkx as nx
import math
from broadcasting import broadcast
import numpy as np
from collections import defaultdict, OrderedDict
from operator import itemgetter
from simulation import load_file_to_graph
import os
from random import shuffle
from random import sample
from itertools import combinations

MAX_NUMBER_OF_FAULT_NODES = 3
NUMBER_OF_NODES_GO_TO_NEXT_SRC = 2 * MAX_NUMBER_OF_FAULT_NODES + 1


def load_file_to_greedy_graph(file_list):
    current_src_id = 0
    sub_graph_list = []
    sub_graph_src_id = []

    threshold_parameters = []
    for file in file_list:
        threshold = get_random_radius_threshold(file)
        threshold_parameters.append(threshold)

    links = []

    total_graph = None

    for i, file in enumerate(file_list):
        # Now we are going to concat graph from a file

        # record the src id
        sub_graph_src_id.append(current_src_id)

        # Now start building the graph
        if i == 0:
            total_graph, append_sub_graph = concat_two_graph(current_src_id, file)

        # Other general Cases
        else:
            total_graph, append_sub_graph = concat_two_graph(current_src_id, file, links, total_graph)

        # Append the subgraph to our list
        sub_graph_list.append(append_sub_graph)

        links = random_generate_link(current_src_id, len(total_graph.nodes))

        current_src_id = len(total_graph.nodes)

    return total_graph, sub_graph_list, set(sub_graph_src_id), threshold_parameters


# Generate the links to the next graph src
def random_generate_link(src_id, size):
    link_candidates = list(range(src_id, size))
    return sample(link_candidates, NUMBER_OF_NODES_GO_TO_NEXT_SRC)


# Return the graph that concat together also return the current graph that is concating to the previous graph
def concat_two_graph(current_src_id, current_sub_graph_file_name, manual_link=None, prev_src_graph=None):
    # Means no prev_graph, just return the graph loaded from file
    if prev_src_graph is None:
        current_sub_graph = pickle.load(open(current_sub_graph_file_name, "rb"))
        # print("Ratio", len(current_sub_graph.edges) / 300)
        return current_sub_graph, current_sub_graph
    else:
        current_sub_graph = pickle.load(open(current_sub_graph_file_name, "rb"))
        # print("Ratio", len(current_sub_graph.edges) / 300)

        # 1. Need to move the id for current_sub_graph
        mapping = dict()
        for i in range(len(current_sub_graph.nodes)):
            mapping[i] = i + current_src_id
        # Relabel the node id with the corresponding id
        nx.relabel_nodes(current_sub_graph, mapping, copy=False)

        # 2. Compose the graph
        concat_graph = compose(prev_src_graph, current_sub_graph)

        # 3. Build the link
        for i in range(len(manual_link)):
            concat_graph.add_edge(current_src_id, manual_link[i])

    return concat_graph, current_sub_graph


# Compose the Two graphs together but are now disconnected component
def compose(src_graph, sub_graph):
    return nx.compose(src_graph, sub_graph)


# Get the Threshold of the geometric graph from file Name
def get_random_radius_threshold(file_name):
    count = 0
    start_idx = 0
    end_idx = 0

    for i in range (len(file_name)):
        if file_name[i] == '_':
            count += 1
            if count == 4:
                start_idx = i + 1
            elif count == 5:
                end_idx = i
                return float(file_name[start_idx : end_idx])
    return 0.0


def remove_greedy_subview_uniform(G, graph_list, t_node_set, t_node_num_list, threshold_parameter):
    return


def remove_greedy_subview_ratio(G, graph_list, t_node_set, t_node_num_list, threshold_parameters):
    return


def remove_greedy_globalview(G, t_node_set, t_node_num_list, threshold_parameters):
    result_dict = dict()

    threshold_to_remove = 0

    for para in threshold_parameters:
        threshold_to_remove += para

    threshold_to_remove = (threshold_to_remove / 2) / (len(threshold_parameters))

    for num_trusted in t_node_num_list:
        trusted_nodes = pick_trusted_centrality_closseness(G, num_trusted, t_node_set, threshold_to_remove)
        result_dict[num_trusted] = trusted_nodes
    return result_dict


def pick_trusted_centrality_closseness(G, num_trusted, t_node_set, threshold_to_remove):
        centrality_dict = nx.closeness_centrality(G)

        centrality_dict = remove_src(centrality_dict, t_node_set)

        centrality_dict = order_dict(centrality_dict)

        # This can be reused once we removing all the neighors in the first round
        throw_away_dict = dict()

        return_trusted_nodes_list = []
        while num_trusted > 0:
            num_trusted = remove_trusted(G, centrality_dict, throw_away_dict, num_trusted, threshold_to_remove, return_trusted_nodes_list)
            centrality_dict = order_dict(throw_away_dict)
            throw_away_dict = dict()

        return return_trusted_nodes_list


# Remove the src from being a candidate for trusted node
def remove_src(centrality_dict, t_node_set):
    for src_id in t_node_set:
        del centrality_dict[src_id]
    return centrality_dict


def remove_trusted(G, centrality_dict, throw_away_dict, number_of_added, threshold_to_remove, return_trusted_nodes_list):
    for id in centrality_dict.keys():
        if number_of_added <= 0 or len(centrality_dict) <= 0:
            break
        if id in throw_away_dict:
            continue
        remove_trusted_helper(G, id, threshold_to_remove, centrality_dict, throw_away_dict)
        return_trusted_nodes_list.append(id)
        number_of_added -= 1
    return number_of_added


def calculate_distance(G, src_id, dst_id):
    src_x, src_y = G.node[src_id]['pos']
    des_x, des_y = G.node[dst_id]['pos']
    distance = math.sqrt(math.pow(src_x - des_x, 2) + math.pow(src_y - des_y, 2))
    return distance


# Remove the potential trusted given current node id
def remove_trusted_helper(G, id, threshold_to_remove, centrality_dict, throw_away_dict):
    edges = nx.edges(G, id)
    delete_ids = []
    for t in edges:
        nei = t[1]
        path_length = calculate_distance(G, id, nei)
        # Remove the nodes to be potential trusted nodes
        if path_length < threshold_to_remove:
            # if id == 141:
            #     print(nei, path_length)
            if nei in centrality_dict.keys():
                delete_ids.append(nei)

    copy_dict = dict(centrality_dict)
    centrality_dict = dict()
    delete_ids = set(delete_ids)

    for id, value in copy_dict.items():
        if id in delete_ids:
            throw_away_dict[id] = value
        else:
            centrality_dict[id] = value

    centrality_dict = order_dict(centrality_dict)
    throw_away_dict = order_dict(throw_away_dict)


def order_dict(centrality_dict):
    centrality_dict = OrderedDict(sorted(centrality_dict.items(), key=itemgetter(1), reverse=True))
    return centrality_dict


if __name__ == '__main__':
    pass
    file_name = "/Users/yingjianwu/Desktop/broadcast/Broadcast_py/subgraphs/geo_300/geo_300_0.15_1"
    graph, sub_graph_list, sub_graph_src_id, threshold_parameters = load_file_to_greedy_graph([file_name])
    print(remove_greedy_globalview(graph, sub_graph_src_id, [270], threshold_parameters))
