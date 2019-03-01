import networkx as nx
from itertools import combinations
from collections import defaultdict, deque
from pickle import dump, load
from time import time

"""
    Configurations: specify TOTAL_NODES, MAX_FAULT_NODES, GRAPH_TYPE, and some other parameters
"""

TOTAL_NODES = 30
MAX_FAULT_NODES = 3

VALID_GRAPH_TYPES = ['Erdos_Renyi', 'Geometric']
GRAPH_TYPE = VALID_GRAPH_TYPES[1]

WANT_WORST_CASE_GRAPH = True  # if we want more rounds before finishing, currently not implemented
ERDOS_RENYI_EDGE_FACTOR = 18  # total num of edges = TOTAL_NODES * edge_factor
GEOMETRIC_THRESHOLD = 0.5


def build_graph(graph_type: str) -> nx.Graph:
    """
    Generates a graph with exactly one connected component of the type specified
    :param graph_type: an entry in VALID_GRAPH_TYPES
    :return: a potentially desired graph, i.e. CPA is correct in this graph under the f-local fault model.
    """

    try_counter, max_tries = 0, 1000
    G, ncc = None, 0
    while ncc != 1:
        try_counter += 1
        if try_counter > max_tries: raise Exception("inappropriate parameters")
        if graph_type == VALID_GRAPH_TYPES[0]:
            total_edges = TOTAL_NODES * ERDOS_RENYI_EDGE_FACTOR
            G = nx.gnm_random_graph(TOTAL_NODES, total_edges)
        elif graph_type == VALID_GRAPH_TYPES[1]:
            G = nx.random_geometric_graph(TOTAL_NODES, GEOMETRIC_THRESHOLD)
        else:
            raise Exception("Invalid graph type")
        ncc = nx.number_connected_components(G)
    assert ncc == 1

    if WANT_WORST_CASE_GRAPH:
        pass
        # TODO: in the original impl, we made sure there are exactly 2f + 1 good nodes around the source
    return G


def check_graph(G: nx.Graph) -> bool:
    """
    We check whether the graph is desired, i.e. CPA is correct in this graph under the f-local fault model,
    in synchronous network by simulating CPA (using breadth-first search) for all combinations of faulty nodes.
    Theorem II indicates when this method exits, all non-faulty nodes receive the fault-free source's
    committed value if and only if the graph is desired.
    :param G: a graph generated by build_graph()
    :return: a boolean indicates whether this graph is desired
    """

    ncc = nx.number_connected_components(G)
    if ncc != 1: return False

    # TODO early termination... but the original implementation remove edges when checking a graph

    nodes_except_src = set(range(1, TOTAL_NODES))
    desired_non_faulty_commits_count = TOTAL_NODES - MAX_FAULT_NODES

    # for each simulation instance, we select a set of faulty nodes
    for fault_nodes in combinations(nodes_except_src, MAX_FAULT_NODES):
        fault_nodes = set(fault_nodes)
        non_faulty_commits_count = broadcast(G, fault_nodes)
        if non_faulty_commits_count != desired_non_faulty_commits_count:
            return False
    return True


def broadcast(G: nx.Graph, faulty_nodes: set, trusted_nodes=set(), need_gui=False):
    """
    Given a potentially desired graph and a set of faulty nodes,
    do a broadcasting to see whether all non-faulty nodes are able to receive the source's commit
    :param G: a graph generated by build_graph()
    :param faulty_nodes: a set of faulty nodes picked by doing combination
    :return: the number of non-faulty nodes that have committed the message, including the source
    """

    """
        Round 0: the source commits (omit appending non_faulty_commit_queue)
    """

    curr_round = 0
    # to record whether a node has committed the value before (due to cyclic graph),
    # if it is the source, or
    # after receiving a commit from the source/trusted node, or
    # after receiving (MAX_FAULT_NODES + 1) commits from incoming nodes
    non_faulty_has_committed = {0}

    # every non-faulty node (except the src) appears in non_faulty_commit_queue only once,
    # but all faulty nodes will not be in this queue
    non_faulty_commit_queue = deque()

    # to record the number of proposes a node receives if it's not directly linked to the source
    propose_received = defaultdict(lambda: 0)

    """
        Round 1: all non-faulty nodes that have incoming edges from the source commit
    """
    curr_round += 1
    src_neis = [edge[1] for edge in G.edges(0)]  # all outgoing neighbours of the source node
    for nei in src_neis:
        if nei not in faulty_nodes:
            non_faulty_commit_queue.append(nei)
            non_faulty_has_committed.add(nei)

    while len(non_faulty_commit_queue):
        curr_round += 1  # for debugging and GUI directory
        for curr_node in range(len(non_faulty_commit_queue)):
            curr_node = non_faulty_commit_queue.popleft()
            curr_node_neis = [edge[1] for edge in G.edges(curr_node)]  # all outgoing neighbours of the current node

            """
                For round >= 2, all non-faulty nodes commits
            """
            for nei in curr_node_neis:
                # If this node has committed before (due to cyclic graph), ignore it.
                # If this node is faulty, ignore it
                if nei in non_faulty_has_committed or nei in faulty_nodes:
                    continue

                # TODO: trusted nodes
                # If this node is non-faulty,
                # it commits iff it has heard (MAX_FAULT_NODES + 1) non-faulty proposes
                # note: faulty nodes don't propose values
                propose_received[nei] += 1
                if propose_received[nei] >= MAX_FAULT_NODES + 1:
                    non_faulty_commit_queue.append(nei)
                    non_faulty_has_committed.add(nei)
    return len(non_faulty_has_committed)


def save_graph(G: nx.Graph, graph_type):
    graph_name = f'n_{TOTAL_NODES}_f_{MAX_FAULT_NODES}_'
    if graph_type == VALID_GRAPH_TYPES[0]:
        graph_name += f'erd_ef_{ERDOS_RENYI_EDGE_FACTOR}_{int(time())}.pi'
    elif graph_type == VALID_GRAPH_TYPES[1]:
        graph_name += f'geo_th_{GEOMETRIC_THRESHOLD}_{int(time())}.pi'

    dump(G, open(graph_name, "wb"))


if __name__ == '__main__':
    pass
    """
        Generate a desired graph
    """
    # G = build_graph(GRAPH_TYPE)
    # while not check_graph(G):
    #     G = build_graph(GRAPH_TYPE)
    # save_graph(G, GRAPH_TYPE)

    """
        or Check whether a graph is valid
    """
    # graph_path = '/Users/haochen/Desktop/Broadcast_py/n_30_f_3_geo_th_0.5_1551466966.pi'
    # G = load(open(graph_path, "rb"))
    # print(check_graph(G))
    # TODO: in addition to this ProofByBroadcasting, do a quick ProofByPartitioning

