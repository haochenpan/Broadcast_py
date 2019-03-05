from enum import Enum
from pickle import load, dump
import os
import networkx as nx
from broadcasting import broadcast
import numpy as np
from collections import defaultdict
from simulation import load_file_to_graph


class Algorithm(Enum):
    UNIFORM_CHOOSE_GLOBAL = 0  # P(a node to be chosen) = 1 / # of total nodes in G
    UNIFORM_CHOOSE_LOCAL = 1  # P(a node to be chosen) = (1 / len(g_list)) * (1 / # of total nodes in g)
    WEIGHTED_EDGES_PROB = 2  # P(a node to be chosen) = (# of its edges / 2) / # of total edges in G
    WEIGHTED_EDGES_RANK = 3  # trusted nodes are choose from nodes with max num of edges to min num of edges


def pick_trusted_nodes(G, graph_list, t_node_set, algorithm, t_node_num_list):
    """
    Pick a set of trusted nodes based on the algorithm chosen
    :param G: the graph to perform simulation
    :param graph_list: a list of graphs (i.e. "subgraphs") that used in generating the graph G
    :param t_node_set: a set of trusted nodes, i.e. original source nodes of subgraphs
    :param algorithm: an element in class Algorithm
    :param t_node_num_list: a list of integers, each entry is the number of new trusted nodes to be chosen
    :return: a dictionary that represents different sets of trusted nodes chosen based on t_node_num_list (see below)
    """
    params_dict = defaultdict(lambda: set())  # key: an entry of t_node_num_list; value: a set of trusted nodes

    # its hard to figure out a list of probabilities that does not incl t_node_set but add up to 1
    # so here we choose an element by element, then test whether we have desired number of nodes
    def probability_choose(new_pick_num, prob):
        t_nodes = t_node_set.copy()
        if new_pick_num > len(G.nodes) - len(t_nodes): raise Exception("Too many trusted nodes")

        while len(t_nodes) < new_pick_num + len(t_node_set):
            t_node = np.random.choice(range(len(G.nodes)), 1, replace=True, p=prob)
            t_node = list(t_node)[0]
            t_nodes.add(t_node)
        return t_nodes

    if algorithm is Algorithm.UNIFORM_CHOOSE_GLOBAL:
        potential_nodes = set(range(len(G.nodes))) - t_node_set
        for t_node_num in t_node_num_list:
            new_trusted_nodes = np.random.choice(list(potential_nodes), t_node_num, replace=False)
            params_dict[t_node_num].update(t_node_set.copy())
            params_dict[t_node_num].update(new_trusted_nodes)

    elif algorithm is Algorithm.UNIFORM_CHOOSE_LOCAL:
        # in order to choose new trusted nodes，builds up a list of probabilities here
        probabilities = []
        for graph in graph_list:
            prob_of_graph = [(1 / len(graph_list)) * (1 / len(graph.nodes)) for n in graph.nodes]
            probabilities.extend(prob_of_graph)

        for t_nodes_num in t_node_num_list:
            trusted_nodes = probability_choose(t_nodes_num, probabilities)
            params_dict[t_nodes_num].update(trusted_nodes)

    elif algorithm is Algorithm.WEIGHTED_EDGES_PROB:
        # the probability of {a node is chosen} = ((# of edges / 2) / total # of edges)
        probabilities = list(map(lambda n: (len(G.edges(n)) / 2) / G.number_of_edges(), G.nodes))

        for t_nodes_num in t_node_num_list:
            trusted_nodes = probability_choose(t_nodes_num, probabilities)
            params_dict[t_nodes_num].update(trusted_nodes)

    elif algorithm is Algorithm.WEIGHTED_EDGES_RANK:
        edges_num_list = [len(G.edges(n)) for n in G.nodes]  # each entry is the num of edges of a node

        # each entry = (num of edges of node i, the index of node i),
        edges_num_rank = sorted((e, i) for i, e in enumerate(edges_num_list))

        for t_nodes_num in t_node_num_list:
            if t_nodes_num > len(G.nodes) - len(t_node_set): raise Exception("Too many trusted nodes")
            new_trusted_nodes = t_node_set.copy()
            # the highest rank, i.e. we want to find the node with max num of edges
            t_node_rank = len(edges_num_rank) - 1

            while len(new_trusted_nodes) < len(t_node_set) + t_nodes_num:
                new_t_node = edges_num_rank[t_node_rank][1]
                new_trusted_nodes.add(new_t_node)
                t_node_rank -= 1
            params_dict[t_nodes_num].update(new_trusted_nodes)

    return params_dict


def simulation(G, params_dict, result_dict):
    print("*** a round of sim started ***")
    for num_of_trusted_nodes, trusted_nodes_set in params_dict.items():
        num_of_commits, num_of_rounds = broadcast(G, faulty_nodes=set(), trusted_nodes=trusted_nodes_set)
        result_dict[num_of_trusted_nodes].append(num_of_rounds)


def simulation_batch():
    g_path_1 = 'n_300_f_3_geo_th_0.15_1.pi'
    g_path_2 = 'n_300_f_3_geo_th_0.25_1.pi'
    g_path_3 = 'test_400_0.14_node_geo'
    G, g_list, t_set = load_file_to_graph([g_path_1, g_path_2, g_path_3])
    # TODO: params_dict should be created many times only for probabilistic algorithm
    params_dict_1 = pick_trusted_nodes(G, g_list, t_set, Algorithm.WEIGHTED_EDGES_RANK, [6, 12, 18])
    result_dict = defaultdict(lambda: [])
    simulation(G, params_dict_1, result_dict)
    for k, v in result_dict.items():
        print(k, v)
    return result_dict


if __name__ == '__main__':
    simulation_batch()
