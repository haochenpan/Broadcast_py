import networkx as nx
from collections import deque
from random import *
import matplotlib.pyplot as plt
import pickle
from broadcasting import broadcast, broadcast_for_gui

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

DIRECTORY = "/Users/haochen/Desktop/Broadcast_py/subgraphs/"


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


def generate_good_nodes_id(current_src_id, total_nodes_count, local_bad_nodes, num_trusted_generate):
    # Need to exclude src and the local bad nodes
    candidate_list = list(range(current_src_id + 1, total_nodes_count))
    trusted_node_candidates = [x for x in candidate_list if x not in local_bad_nodes]
    return sample(trusted_node_candidates, num_trusted_generate)


def concat_two_graph(current_src_id, current_sub_graph_file_name, manual_link=None, prev_src_graph=None):
    # Means no prev_graph, just return the graph loaded from file
    if prev_src_graph is None:
        current_sub_graph = pickle.load(open(DIRECTORY + current_sub_graph_file_name, "rb"))
        # print("Ratio", len(current_sub_graph.edges) / 300)
        return current_sub_graph
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
    return concat_graph


# Main function for concatinating graphs and run the broadcast
# Note: fault nodes are not arbitrary
# Note: all the previous source node will act like a trusted node
def concat_graph_main(added_nodes_list):
    current_src_id = 0
    concat_fault_list = []

    # key: number of trusted_nodes in total graph
    # value: list containing the id of those trusted_nodes
    trusted_nodes_dict = dict()
    for added_nodes_id in added_nodes_list:
        trusted_nodes_dict[added_nodes_id] = TRUSTED_SRC_NODES[:]

    links = []

    for i in range(len(List_OF_FILE_NAME)):
        # Now we are going to concat graph from a file
        current_file_name = List_OF_FILE_NAME[i]

        # Now start building the graph
        if i == 0:
            total_graph = concat_two_graph(current_src_id, current_file_name)

        # Other general Cases
        else:
            total_graph = concat_two_graph(current_src_id, current_file_name, links, total_graph)

        # Node number after a new concatination
        current_total_nodes = len(total_graph.nodes)

        # Added bad nodes in local graph
        local_bad_nodes = []
        # local_bad_nodes = generate_bad_nodes_id(current_src_id, current_total_nodes)
        concat_fault_list.extend(local_bad_nodes)

        # We do this because we want the different nodes parameter testing against the same graph
        for additional_trusted_nodes_count in added_nodes_list:
            # Add trusted_nodes in local graph
            if STRATEGY == 0:
                local_trusted_nodes_count = additional_trusted_nodes_count // len(TRUSTED_SRC_NODES)

            elif STRATEGY == 1:
                local_trusted_nodes_count = int((additional_trusted_nodes_count // TOTAL_RATIO) * RATIO_LIST[i])

            else:
                local_trusted_nodes_count = 0

            local_trusted_nodes_list = generate_good_nodes_id(current_src_id, current_total_nodes,
                                                              local_bad_nodes, local_trusted_nodes_count)

            # print(additional_trusted_nodes_count, local_trusted_nodes_list)
            prev_good_list = trusted_nodes_dict[additional_trusted_nodes_count]
            prev_good_list.extend(local_trusted_nodes_list)
            trusted_nodes_dict[additional_trusted_nodes_count] = prev_good_list

        # Generate the links to link with the next graph

        links = random_generate_link(current_src_id, len(total_graph.nodes))
        # the new added src_id is always the previous graph last index + 1, which is the old_total_graph length
        current_src_id = len(total_graph.nodes)

    # Note that you need to include all the subgraph src nodes
    # print(concat_trusted_list)
    # print(concat_fault_list)

    return total_graph, set(concat_fault_list), trusted_nodes_dict


# Note that rounds = number of graph generated
def uni_batch_running(rounds=5000):
    # 12, 24, 48, 96, 188
    uni_round_dict = dict()
    good_nodes_test_union = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280]

    for para in good_nodes_test_union:
        uni_round_dict[para] = []

    for i in range(rounds):
        # Generate the graph
        graph, fault_nodes, concat_good_dict = concat_graph_main(good_nodes_test_union)
        # Tested against the parameter
        for para in good_nodes_test_union:
            good_nodes = concat_good_dict[para]
            total_commit, total_round = broadcast(graph, fault_nodes, good_nodes)
            uni_round_dict[para].append(total_round)

        pickle.dump(graph, open("uni_data_1500_graph.p", "wb"))

    pickle.dump(uni_round_dict, open("uni_data_node_1500_5000_base_times", "wb"))


def ratio_batch_running(rounds=5000):
    global STRATEGY
    STRATEGY = 1
    ratio_round_dict = dict()
    good_nodes_test_ratio = [0, 9, 18, 36, 72, 144, 288, 576, 864]

    for para in good_nodes_test_ratio:
        ratio_round_dict[para] = []

    for i in range(rounds):
        graph, fault_nodes, concat_good_dict = concat_graph_main(good_nodes_test_ratio)
        for para in good_nodes_test_ratio:
            good_nodes = concat_good_dict[para]
            total_commit, total_round = broadcast(graph, fault_nodes, good_nodes)
            ratio_round_dict[para].append(total_round)
    pickle.dump(ratio_round_dict, open("ratio_data_node_1500_5000_base_times", "wb"))


def main():
    uni_batch_running()
    ratio_batch_running()
    # dic = pickle.load(open("uni_data5000_1000_times", "rb"))
    # for k, v in dic.items():
    #     print(k, v)
    # print(dic)


if __name__ == '__main__':
    main()
