"""
    Find a desired graph G(V, E) by checking all valid partitions (L, R, F) satisfy:
        L -> R  OR {Ns+ union R} is not empty.

    Note: a valid partition (L, R, F) means:
        the source is in L AND
        R is not empty AND
        F is a feasible f-local set

    Note: This method of checking a graph is EXTREMELY SLOW, .
"""

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import pickle

NUM_OF_NODES = 10
F_LOCAL = 3


def gen_some_graph():
    G = nx.random_geometric_graph(NUM_OF_NODES, 0.5)
    ncc = nx.number_connected_components(G)
    while ncc != 1:
        G = nx.random_geometric_graph(NUM_OF_NODES, 0.5)
        ncc = nx.number_connected_components(G)
    return G


def get_neighbours(G):
    S = defaultdict(lambda: set())
    for node in range(NUM_OF_NODES):
        S[node] = set(G.neighbors(node))
    return S


def graph_it(G):
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, nodelist=G.nodes, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes,
                           node_size=80, )
    nx.draw_networkx_labels(G, pos)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis('off')
    plt.show()


def intersect_gt_than(S1, S2, cap):
    # iterate the shorter set
    # return a boolean indicates whether |intersection| > cap
    if len(S2) < len(S1):
        S1, S2 = S2, S1

    cnt = 0
    for s in S1:
        if s in S2:
            cnt += 1
            if cnt > cap:
                return True
    return False


def is_ffs(S, LR, F):
    if intersect_gt_than(S[0], F, F_LOCAL): return False
    for node in LR:
        if intersect_gt_than(S[node], F, F_LOCAL): return False
    return True


def partition_gen(S):
    nodes_except_source = set(range(1, NUM_OF_NODES))
    # 0 <= num_in_F <= |nodes_except_source| - 1 since the source is not in F, and R is not empty
    for num_in_F in range(len(nodes_except_source)):
        print(f"current num_in_F={num_in_F}")
        for F_partition in combinations(nodes_except_source, num_in_F):
            F_partition = set(F_partition)
            L_R_partition = nodes_except_source - F_partition
            if not is_ffs(S, L_R_partition, F_partition): continue

            # 0 <= num_in_L_except_source <= len(L_R_partition) - 1 to ensure R is not empty
            for num_in_L_except_source in range(len(L_R_partition)):
                for L_partition in combinations(L_R_partition, num_in_L_except_source):
                    L_partition = set(L_partition)
                    R_partition = L_R_partition - L_partition
                    yield L_partition, R_partition


def is_partition_desired(S, L, R):
    if intersect_gt_than(S[0], R, 0): return True
    for node in R:
        if intersect_gt_than(S[node], L, F_LOCAL): return True
    return False


def get_desired_graph():
    result = None
    while result is None:
        G = gen_some_graph()
        S = get_neighbours(G)
        for L, R in partition_gen(S):
            if not is_partition_desired(S, L, R):
                break
        else:
            result = G
            graph_it(result)
    return result


def check_desired_graph(G):
    global NUM_OF_NODES
    NUM_OF_NODES = len(G.nodes)
    S = get_neighbours(G)
    print("graph loaded")
    print(f"num of nodes: {NUM_OF_NODES}")
    for L, R in partition_gen(S):
        if not is_partition_desired(S, L, R):
            print(L, R)
            print("graph not desired")
            return
    print("graph is desired")


if __name__ == '__main__':
    # check_desired_graph(G)
    get_desired_graph()
    # print(get_desired_graph())
