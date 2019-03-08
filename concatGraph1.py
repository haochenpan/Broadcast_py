import pickle
import networkx as nx
from collections import defaultdict, OrderedDict, deque
from operator import itemgetter
import os
from random import shuffle
from random import sample
from itertools import combinations


MAX_NUMBER_OF_FAULT_NODES = 3
NUMBER_OF_NODES_GO_TO_NEXT_SRC = 2 * MAX_NUMBER_OF_FAULT_NODES + 1
rootdir = os.path.join(os.getcwd(), 'subgraphs/')


# total_Graph: Concat_Graph
# sub_graph_list: a list of small graph
# set of good src id
def load_file_to_greedy_graph(file_list):
    current_src_id = 0
    sub_graph_list = []
    sub_graph_src_id = []

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

    return total_graph, sub_graph_list, set(sub_graph_src_id)


# Generate the links to the next graph src
def random_generate_link(src_id, size):
    link_candidates = list(range(src_id, size))
    links = sample(link_candidates, NUMBER_OF_NODES_GO_TO_NEXT_SRC * (MAX_NUMBER_OF_FAULT_NODES + 1 + 1))
    # print(links)
    return links


def buildLink (manual_link, current_src_id, end_graph_id, G):
    candidates = sample(list(range(current_src_id + 1, end_graph_id)), MAX_NUMBER_OF_FAULT_NODES + 1)
    candidates.append(current_src_id)

    for i in range(len(candidates)):
        next_graph_nodes = candidates[i]
        start_link = i * NUMBER_OF_NODES_GO_TO_NEXT_SRC
        end_link = (i + 1) * NUMBER_OF_NODES_GO_TO_NEXT_SRC
        while start_link < end_link :
            edge = manual_link[start_link]
            # print(next_graph_nodes, edge)
            G.add_edge(next_graph_nodes, edge)
            start_link += 1

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

        end_graph_id = current_src_id + len(current_sub_graph.nodes)

        # 3. Build the link
        buildLink(manual_link, current_src_id, end_graph_id, concat_graph)

    return concat_graph, current_sub_graph


# Compose the Two graphs together but are now disconnected component
def compose(src_graph, sub_graph):
    return nx.compose(src_graph, sub_graph)


def main():
    file_list_1 = [
        rootdir + "geo_300/geo_300_0.15_1",

        rootdir + "geo_300/geo_300_0.15_1551538743.pi",

        rootdir + "geo_300/geo_300_0.25_3",

        rootdir + "geo_300/geo_300_0.25_1",

        rootdir + "geo_300/geo_300_0.2_1551741145.pi",
    ]

    file_list_2 = [
        rootdir + "geo_300/geo_300_0.15_1551697859.pi",

        rootdir + "geo_300/geo_300_0.2_1551741145.pi",

        rootdir + "geo_300/geo_300_0.15_1551718271.pi",

        rootdir + "geo_300/geo_300_0.25_2",

        rootdir + "geo_300/geo_300_0.2_1551777530.pi",
    ]

    file_list_3 = [
        rootdir + "geo_300/geo_300_0.15_1551866492.pi",

        rootdir + "geo_300/geo_300_0.25_3",

        rootdir + "geo_300/geo_300_0.2_1551823766.pi",

        rootdir + "geo_300/geo_300_0.2_1551838479.pi",

        rootdir + "geo_300/geo_300_0.15_1551538743.pi",
    ]

    file_list_4 = [
        rootdir + "geo_300/geo_300_0.15_1551866492.pi",

        rootdir + "geo_300/geo_300_0.15_1551873754.pi",

        rootdir + "geo_300/geo_300_0.15_1551538743.pi",

        rootdir + "geo_300/geo_300_0.15_1551746257.pi",

        rootdir + "geo_300/geo_300_0.15_1551697859.pi",
    ]

    file_list_5 = [
        rootdir + "geo_300/geo_300_0.2_1551838479.pi",

        rootdir + "geo_300/geo_300_0.2_1551823766.pi",

        rootdir + "geo_300/geo_300_0.2_1551822029.pi",

        rootdir + "geo_300/geo_300_0.2_1551814639.pi",

        rootdir + "geo_300/geo_300_0.2_1551679183.pi",
    ]

    file_list_6 = [
        rootdir + "geo_300/geo_300_0.25_1551548818.pi",

        rootdir + "geo_300/geo_300_0.25_2",

        rootdir + "geo_300/geo_300_0.25_3",

        rootdir + "geo_300/geo_300_0.25_1551548818.pi",

        rootdir + "geo_300/geo_300_0.25_1551549453.pi",
    ]

    # big_list_file = [file_list_1, file_list_2, file_list_3, file_list_4, file_list_5]
    big_list_file = [file_list_6]
    count = 10
    for file_list in big_list_file:
        G, glist, src_ids = load_file_to_greedy_graph(file_list)
        for g in glist:
            print(nx.number_of_edges(g))
        pickle.dump(G, open(f"Big_Graph_{count}", "wb"))
        count += 1



if __name__ == '__main__':
    pass
    main()
