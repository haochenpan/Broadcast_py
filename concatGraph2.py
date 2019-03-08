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

src_id = [0, 300, 600, 900, 1500]

def loadGraph(pickle_name) :
    G = pickle.load(open(pickle_name, "rb"))
    return G


# Generate the links to the next graph src
def random_generate_link(src_id, size):
    link_candidates = list(range(src_id, size))
    links = sample(link_candidates, NUMBER_OF_NODES_GO_TO_NEXT_SRC * (MAX_NUMBER_OF_FAULT_NODES + 1))
    # print(links)
    return links


def buildLink(manual_link, current_src_id, end_graph_id, G):
    candidates = sample(list(range(current_src_id, end_graph_id)), MAX_NUMBER_OF_FAULT_NODES + 1)
    for i in range(len(candidates)):
        next_graph_nodes = candidates[i]
        start_link = i * NUMBER_OF_NODES_GO_TO_NEXT_SRC
        end_link = (i + 1) * NUMBER_OF_NODES_GO_TO_NEXT_SRC
        while start_link < end_link :
            edge = manual_link[start_link]
            G.add_edge(next_graph_nodes, edge)
            start_link += 1

def addMoreLink(G, src_id_list):
    link_to_index = 2
    while link_to_index < 5:
        link_from_id = 0
        # print(link_to_index)
        while link_from_id < link_to_index - 1:
            current_link_from_id = src_id_list[link_from_id]
            link = random_generate_link(current_link_from_id, current_link_from_id + 300)
            buildLink(link, src_id_list[link_to_index], src_id_list[link_to_index] + 300, G)
            link_from_id += 1
        link_to_index += 1


def main():
    count = 11
    src_id = [0, 300, 600, 900, 1200]

    # graph_name = ["Big_Graph_0", "Big_Graph_1", "Big_Graph_2", "Big_Graph_3", "Big_Graph_4"]
    graph_name = ["Big_Graph_10"]
    for name in graph_name:
        G = loadGraph(name)
        addMoreLink(G, src_id)
        pickle.dump(G, open(f"Big_Graph_{count}", "wb"))
        count += 1





if __name__ == '__main__':
    pass
    main()
