from broadcasting import broadcast
from collections import defaultdict, OrderedDict
from enum import Enum
from itertools import combinations
import networkx as nx
import numpy as np
from operator import itemgetter
import os
from pickle import load, dump
from random import shuffle
from simulation import load_file_to_graph


class Algorithm(Enum):
    DEGREE_CENTRALITY = 0
    EIGEN_CENTRALITY = 1
    CLOSENESS_CENTRALITY = 2
    BETWEENNESS_CENTRALITY = 3

    UNIFORM_CHOOSE_PROB_1 = 4  # P(a node to be chosen) = 1 / # of total nodes in G
    UNIFORM_CHOOSE_PROB_2 = 5  # P(a node to be chosen) = (1 / len(g_list)) * (1 / # of total nodes in g)
    WEIGHTED_EDGES_PROB = 6  # P(a node to be chosen) = (# of its edges / 2) / # of total edges in G


rootdir = os.path.join(os.getcwd(), 'subgraphs')


def batch_load_file_to_graph(num_of_G, num_of_g_per_G,
                             g_type=['geo'], num_of_nodes_per_g=[300],
                             graph_param='all', arrangement='sorted'):
    """
    A generator function that provides an iterator of a (G, [g, ...], {t, ...}) tuple created by load_file_to_graph()
    :param num_of_G: the number of "concated" graphs need to be generated, if = 0, then generates all combinations
    :param num_of_g_per_G: the number of "subgraphs" in a "concated" graph
    :param g_type: the type of the subgraph, e.g. "geo"
    :param num_of_nodes_per_g: e.g. [200, 300]
    :param graph_param: 'all' or e.g. [0.15, 0.2]
    :param arrangement: 'sorted' or 'random'
    :return: a generator that delivers a (G, [g, ...], {t, ...}) tuple
             up until all combination is given / num_of_G has reached
    """

    # find all paths of graphs satisfy filters above
    graph_paths = []
    for root, subdirs, files in os.walk(rootdir):
        files = filter(lambda x: x.split('_')[0] in g_type, files)  # filter graph of that type
        files = filter(lambda x: int(x.split('_')[1]) in num_of_nodes_per_g, files)  # filter graph with that many nodes
        if graph_param != 'all':
            files = filter(lambda x: float(x.split('_')[2]) in graph_param, files)  # filter graph with that param
        files = map(lambda x: os.path.join(root, x), files)  # add full path to the file name
        graph_paths.extend(files)

    # check a possible exception
    if len(graph_paths) < num_of_g_per_G:
        raise Exception(f"subgraphs are not enough, found {len(graph_paths)} but needs at least {num_of_g_per_G}")

    # sort them or shuffle them
    if arrangement == 'sorted':
        graph_paths = sorted(graph_paths)
    elif arrangement == 'random':
        shuffle(graph_paths)

    # the generator function
    if num_of_G == 0:  # if we want all combinations
        for paths in combinations(graph_paths, num_of_g_per_G):
            yield load_file_to_graph(list(paths))
    else:  # if we want up to a number of combinations
        graph_gen_counter = 0
        for paths in combinations(graph_paths, num_of_g_per_G):
            if graph_gen_counter >= num_of_G: return
            yield load_file_to_graph(list(paths))
            graph_gen_counter += 1


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

    # assigns an importance score based purely on the number of links held by each node.
    # finding very connected individuals, popular individuals,
    # individuals who are likely to hold most information or individuals who can quickly connect with the wider network
    def pick_trusted_centrality_edge(graph, num_of_trusted):
        centrality_dict = nx.degree_centrality(graph)
        return centrality_top_k(centrality_dict, num_of_trusted)

    # Like degree centrality, EigenCentrality measures a node’s influence based on
    # the number of links it has to other nodes within the network.
    # EigenCentrality then goes a step further by also taking into account
    # how well connected a node is, and how many links their connections have, and so on through the network.
    # EigenCentrality can identify nodes with influence over the whole network, not just those directly connected to it.
    def pick_trusted_centrality_eigen(graph, num_of_trusted):
        # Max_iter needs to manually set up based on different graph
        try:
            centrality_dict = nx.eigenvector_centrality(graph, max_iter=500)
            return centrality_top_k(centrality_dict, num_of_trusted)
        except Exception as expected:
            print("pick_trusted_centrality_eigen exception raised")
            print(repr(expected))
            return {-1}  # an exception flag

    # measure scores each node based on their ‘closeness’ to all other nodes within the network
    # calculates the shortest paths between all nodes, then assigns each node a score based on its sum of shortest paths
    # For finding the individuals who are best placed to influence the entire network most quickly
    # Closeness centrality can help find good ‘broadcasters’,
    # but in a highly connected network you will often find all nodes have a similar score.
    # What may be more useful is using Closeness to find influences within a single cluster
    # This may not works for geometric graph because geometric is also based on distance
    def pick_trusted_centrality_closseness(graph, num_of_trusted):
        centrality_dict = nx.closeness_centrality(graph)
        return centrality_top_k(centrality_dict, num_of_trusted)

    # Betweenness centrality measures the number of times a node lies on the shortest path between other nodes
    # Not sure
    def pick_trusted_centrality_betweeness(graph, num_of_trusted):
        centrality_dict = nx.betweenness_centrality(graph)
        return centrality_top_k(centrality_dict, num_of_trusted)

    # Pick the top k among the trusted
    def centrality_top_k(centrality_dict, num_of_trusted):
        if num_of_trusted == 0:
            return []

        centrality_dict = OrderedDict(sorted(centrality_dict.items(), key=itemgetter(1), reverse=True))
        for src_id in t_node_set:
            del centrality_dict[src_id]
        good_nodes_list = []

        count = 0
        for k, v in centrality_dict.items():
            count += 1
            good_nodes_list.append(k)
            if count == num_of_trusted:
                break

        return set(good_nodes_list)

    if algorithm is Algorithm.UNIFORM_CHOOSE_PROB_1:
        potential_nodes = set(range(len(G.nodes))) - t_node_set
        for t_node_num in t_node_num_list:
            new_trusted_nodes = np.random.choice(list(potential_nodes), t_node_num, replace=False)
            params_dict[t_node_num].update(t_node_set.copy())
            params_dict[t_node_num].update(new_trusted_nodes)

    elif algorithm is Algorithm.UNIFORM_CHOOSE_PROB_2:
        # in order to choose new trusted nodes，builds up a list of probabilities here
        probabilities = []
        for graph in graph_list:
            prob_of_graph = [(1 / len(graph_list)) * (1 / len(graph.nodes)) for n in graph.nodes]
            probabilities.extend(prob_of_graph)

        for t_node_num in t_node_num_list:
            trusted_nodes = probability_choose(t_node_num, probabilities)
            params_dict[t_node_num].update(trusted_nodes)

    elif algorithm is Algorithm.WEIGHTED_EDGES_PROB:
        # the probability of {a node is chosen} = ((# of edges / 2) / total # of edges)
        probabilities = list(map(lambda n: (len(G.edges(n)) / 2) / G.number_of_edges(), G.nodes))

        for t_node_num in t_node_num_list:
            trusted_nodes = probability_choose(t_node_num, probabilities)
            params_dict[t_node_num].update(trusted_nodes)

    elif algorithm is Algorithm.DEGREE_CENTRALITY:
        for t_node_num in t_node_num_list:
            trusted_nodes = pick_trusted_centrality_edge(G, t_node_num)
            params_dict[t_node_num].update(trusted_nodes)
            params_dict[t_node_num].update(t_node_set.copy())

    elif algorithm is Algorithm.EIGEN_CENTRALITY:
        for t_node_num in t_node_num_list:
            trusted_nodes = pick_trusted_centrality_eigen(G, t_node_num)
            params_dict[t_node_num].update(trusted_nodes)
            params_dict[t_node_num].update(t_node_set.copy())

    elif algorithm is Algorithm.CLOSENESS_CENTRALITY:
        for t_node_num in t_node_num_list:
            trusted_nodes = pick_trusted_centrality_closseness(G, t_node_num)
            params_dict[t_node_num].update(trusted_nodes)
            params_dict[t_node_num].update(t_node_set.copy())

    elif algorithm is Algorithm.BETWEENNESS_CENTRALITY:
        for t_node_num in t_node_num_list:
            trusted_nodes = pick_trusted_centrality_betweeness(G, t_node_num)
            params_dict[t_node_num].update(trusted_nodes)
            params_dict[t_node_num].update(t_node_set.copy())

    return params_dict


def simulation_batch():
    def simulate_rank(algorithm):
        """
        Used in deterministic algorithms 0, 1, 2, and 3. Generates a params dict and run simulation once
        :param algorithm: an entry like Algorithm.DEGREE_CENTRALITY
        :return: no return but updates the result_dict
        """
        params_dict = pick_trusted_nodes(G, g_list, t_set, algorithm, t_nodes_list)
        if algorithm is Algorithm.EIGEN_CENTRALITY and -1 in t_set:  # to handle a special exception of Eigen
            print("excepted detected, append -1 to the list")
            for num_of_t_nodes, t_nodes_set in params_dict.items():
                result_dict[num_of_t_nodes][algorithm.value].append(-1)
        else:
            for num_of_t_nodes, t_nodes_set in params_dict.items():
                num_of_commits, num_of_rounds = broadcast(G, faulty_nodes=set(), trusted_nodes=t_nodes_set)
                result_dict[num_of_t_nodes][algorithm.value].append(num_of_rounds)

    def simulate_prob(algorithm, num_of_sim):
        """
        Used in probalistic algorithms 4, 5, and 6. Generates params dict n times and run n times simulation
        :param algorithm: an entry like Algorithm.UNIFORM_CHOOSE_PROB_1
        :param num_of_sim: number of simulations on that graph with that algorithm
        :return: no return but updates the result_dict
        """
        temp_result_dict = defaultdict(lambda: [])
        for i in range(num_of_sim):
            params_dict = pick_trusted_nodes(G, g_list, t_set, algorithm, t_nodes_list)
            for num_of_t_nodes, t_nodes_set in params_dict.items():
                num_of_commits, num_of_rounds = broadcast(G, faulty_nodes=set(), trusted_nodes=t_nodes_set)
                temp_result_dict[num_of_t_nodes].append(num_of_rounds)

        for num_of_t_nodes, num_of_rounds_list in temp_result_dict.items():
            # print(num_of_rounds_list)
            result_dict[num_of_t_nodes][algorithm.value].append(sum(num_of_rounds_list) / len(num_of_rounds_list))

    # {key: num of trusted; value: {key: algo enum; value: num of rounds}}
    result_dict = defaultdict(lambda: defaultdict(lambda: []))

    # for testing
    # for G, g_list, t_set in batch_load_file_to_graph(1, 5, num_of_nodes_per_g=[300]):
    # n, m = broadcast(G, {}, t_set)
    # print(n, m)

    for G, g_list, t_set in batch_load_file_to_graph(0, 5, num_of_nodes_per_g=[300]):
        print("*** a graph sim has started 0123 ***")

        t_nodes_list = [i for i in range(25)]
        t_nodes_list.extend([i for i in range(25, 276, 25)])

        # if algo 0 - 3:
        simulate_rank(Algorithm.DEGREE_CENTRALITY)
        simulate_rank(Algorithm.EIGEN_CENTRALITY)
        simulate_rank(Algorithm.CLOSENESS_CENTRALITY)
        simulate_rank(Algorithm.BETWEENNESS_CENTRALITY)

        # if algo 4 - 6:
        # num_of_sim_per_graph = 10
        # simulate_prob(Algorithm.UNIFORM_CHOOSE_PROB_1, num_of_sim_per_graph)
        # simulate_prob(Algorithm.UNIFORM_CHOOSE_PROB_2, num_of_sim_per_graph)
        # simulate_prob(Algorithm.WEIGHTED_EDGES_PROB, num_of_sim_per_graph)

    # iterate the default dict to generate a normal dict for serializing
    outside_dict = dict()
    for k, v in result_dict.items():
        inside_dict = dict()
        for k2, v2 in v.items():
            inside_dict[k2] = v2
        outside_dict[k] = inside_dict

    return outside_dict


if __name__ == '__main__':
    pass
    # what to config:
    # 1) batch_load_file_to_graph(num of G wanted (0=all), num of subgraphs / G, other params see doc str above)
    # 2) t_nodes_list
    # 3.1) if simulate_rank(), you are good to go
    # 3.2) if simulate_prob(), i.e. algo 4, 5, and 6: num_of_sim_per_graph
    # 4) file name below
    print("simulation started")
    result_dict = simulation_batch()
    with open("result_dict_300x5_0123.pickle", "wb") as output_file:
        dump(result_dict, output_file)
    print("simulation finished")

    # with open("result_dict_300x5_456.pickle", "rb") as input_file:
    #     result_dict = load(input_file)
    #     for k, v in result_dict.items():
    #         print(k)
    #         for k2, v2 in v.items():
    #             print(k2, v2)
