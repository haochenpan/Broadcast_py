import networkx as nx
from collections import deque
from random import *
import matplotlib.pyplot as plt
import pickle
from broadcasting import broadcast, broadcast_for_gui
from collections import OrderedDict
from operator import itemgetter

###################################################
# Configuration Variable
###################################################

MAX_NUMBER_OF_FAULT_NODES = 3

NUMBER_OF_NODES_GO_TO_NEXT_SRC = 2 * MAX_NUMBER_OF_FAULT_NODES + 1

# List of fileName that you want to concat together
List_OF_FILE_NAME = ["n_300_f_3_geo_th_0.15_1.pi", "n_300_f_3_geo_th_0.25_1.pi",
                     "n_300_f_3_geo_th_0.25_2.pi", "test_300_0.2_node_geo",
                     "test_300_0.17_node_geo"]

TRUSTED_SRC_NODES = [0, 300, 600, 900, 1200]

# 0: Indicates HAOCHEN's strategy, which is the uniform distributed trusted nodes
# 1: Idicates Steven's strategy, which is distributed trusted nodes based on Ratio
STRATEGY = 0

# Parameters used in Strategy 1
# RATIO_LIST = [1, 4, 3, 2, 2, 4]
# # TOTAL_RATIO = 16

RATIO_LIST = [3, 1, 1, 2, 2]
TOTAL_RATIO = 9

DIRECTORY = "/Users/yingjianwu/Desktop/broadcast/Broadcast_py/subgraphs/"


#################################################
# Function to concat graphs
#################################################


# Check if the vertex is a faulty node
def is_fault_node(vertex, fault_nodes):
    return vertex in fault_nodes


# Compose the Two graphs together but are now disconnected component
def compose(src_graph, sub_graph):
    return nx.compose(src_graph, sub_graph)


# Generate the links to the next graph src
def random_generate_link(src_id, size):
    link_candidates = list(range(src_id, size))
    return sample(link_candidates, NUMBER_OF_NODES_GO_TO_NEXT_SRC)


def generate_bad_nodes_id(src_id, total_nodes_count):
    # +1 because src_id is always trusted
    fault_candidate_list = list(range(src_id + 1, total_nodes_count))
    return sample(fault_candidate_list, MAX_NUMBER_OF_FAULT_NODES)


# Return the graph that concat together also return the current graph that is concating to the previous graph
def concat_two_graph(current_src_id, current_sub_graph_file_name, manual_link=None, prev_src_graph=None):
    # Means no prev_graph, just return the graph loaded from file
    if prev_src_graph is None:
        current_sub_graph = pickle.load(open(DIRECTORY + current_sub_graph_file_name, "rb"))
        # print("Ratio", len(current_sub_graph.edges) / 300)
        return current_sub_graph, current_sub_graph
    else:
        current_sub_graph = pickle.load(open(DIRECTORY + current_sub_graph_file_name, "rb"))
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


# Main function for concatinating graphs and run the broadcast
# Note: fault nodes are not arbitrary
# Note: all the previous source node will act like a trusted node
def load_file_to_graph(file_list):
    current_src_id = 0
    sub_graph_list = []
    sub_graph_src_id = []

    links = []

    total_graph = None

    for i in range(len(file_list)):
        # Now we are going to concat graph from a file
        current_file_name = List_OF_FILE_NAME[i]

        # record the src id
        sub_graph_src_id.append(current_src_id)

        # Now start building the graph
        if i == 0:
            total_graph, append_sub_graph = concat_two_graph(current_src_id, current_file_name)

        # Other general Cases
        else:
            total_graph, append_sub_graph = concat_two_graph(current_src_id, current_file_name, links, total_graph)

        # Append the subgraph to our list
        sub_graph_list.append(append_sub_graph)

        links = random_generate_link(current_src_id, len(total_graph.nodes))

        current_src_id = len(total_graph.nodes)

    return total_graph, sub_graph_list, sub_graph_src_id


def pick_trusted_centrality_edge(graph, num_of_trusted):
    centrality_dict = nx.degree_centrality(graph)
    return centrality_top_k(centrality_dict, num_of_trusted)


def pick_trusted_centrality_eigen(graph, num_of_trusted):
    # Max_iter needs to manually set up based on different graph
    centrality_dict = nx.eigenvector_centrality(graph, max_iter=500)
    return centrality_top_k(centrality_dict, num_of_trusted)


def pick_trusted_centrality_closseness(graph, num_of_trusted):
    centrality_dict = nx.closeness_centrality(graph)
    return centrality_top_k(centrality_dict, num_of_trusted)


def pick_trusted_centrality_betweeness(graph, num_of_trusted):
    centrality_dict = nx.betweenness_centrality(graph)
    return centrality_top_k(centrality_dict, num_of_trusted)


# Pick the top k among the trusted
def centrality_top_k(centrality_dict, num_of_trusted):
    if num_of_trusted == 0:
        return []

    centrality_dict = OrderedDict(sorted(centrality_dict.items(), key=itemgetter(1), reverse=True))
    good_nodes_list = []

    count = 0
    for k, v in centrality_dict.items():
        count += 1
        good_nodes_list.append(k)
        if count == num_of_trusted:
            break

    return good_nodes_list


def main():
    graph = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/subgraphs/n_300_f_3_geo_th_0.15_1.pi","rb"))
    print(pick_trusted_centrality_eigen(graph, 3))


if __name__ == '__main__':
    main()

