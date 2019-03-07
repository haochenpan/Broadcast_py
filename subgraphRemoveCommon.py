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
rootdir = os.path.join(os.getcwd(), 'subgraphs')
print(rootdir)


# 1. Closseness 2. Betweeness, 3 Edges
def remove_greedy(G, t_node_set, t_node_num_list, alg_num, common_threshold = 3):
    result_dict = dict()
    for num_trusted in t_node_num_list:
        trusted_nodes = pick_trusted_centrality(G, num_trusted, t_node_set, common_threshold, alg_num)
        result_dict[num_trusted] = trusted_nodes
        result_dict[num_trusted].update(t_node_set.copy())
    return result_dict


def pick_trusted_centrality(G, num_trusted, t_node_set, overlap, alg_num):
    if alg_num == 1:
        centrality_dict = nx.closeness_centrality(G)
    elif alg_num == 2:
        centrality_dict = nx.betweenness_centrality(G)
    elif alg_num == 3:
        centrality_dict = nx.degree_centrality(G)

    centrality_dict = remove_src(centrality_dict, t_node_set)

    centrality_dict = order_dict(centrality_dict)
    return_trusted_nodes = get_trusted(G, centrality_dict, num_trusted, overlap)

    return return_trusted_nodes


# Remove the src from being a candidate for trusted node
def remove_src(centrality_dict, t_node_set):
    for src_id in t_node_set:
        del centrality_dict[src_id]
    return centrality_dict


def get_trusted(G, centrality_dict: dict, num_trusted: int, overlap: int):
    return_trusted_nodes = set()

    while num_trusted > 0:
        copy_dict = order_dict(dict(centrality_dict))

        centrality_dict = dict()

        for id in copy_dict.keys():
            # Check if this id can be picked as trusted based on threshold
            if checkTrusted(G, id, return_trusted_nodes, overlap):
                return_trusted_nodes.add(id)
                num_trusted -= 1
                if num_trusted == 0:
                    return return_trusted_nodes

        for node, value in copy_dict.items():
            if node not in return_trusted_nodes:
                centrality_dict[node] = value
        overlap = overlap * 2

    return return_trusted_nodes


# Check if the current node has overlap more than the threshold
def checkTrusted(G, id, return_trusted_nodes, overlap):
    for trusted_id in return_trusted_nodes:
        common_neis = len(list(nx.common_neighbors(G, id, trusted_id)))
        if common_neis > overlap:
            return False
    return True


def order_dict(centrality_dict):
    centrality_dict = OrderedDict(sorted(centrality_dict.items(), key=itemgetter(1), reverse=True))
    return centrality_dict


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
            yield load_file_to_greedy_graph(list(paths))
    else:  # if we want up to a number of combinations
        graph_gen_counter = 0
        for paths in combinations(graph_paths, num_of_g_per_G):
            if graph_gen_counter >= num_of_G: return
            yield load_file_to_greedy_graph(list(paths))
            graph_gen_counter += 1


def simulation_batch():
    # {key: num of trusted; value: {key: algo enum; value: num of rounds}}
    result_dict = defaultdict(lambda: defaultdict(lambda: []))
    for G, g_list, t_set in batch_load_file_to_graph(0, 1, num_of_nodes_per_g=[300]):
        print("*** a graph sim has started ***")
        t_nodes_list = [i for i in range(10)]
        t_nodes_list.extend([i for i in range(30, 271, 30)])

        params_dict = remove_greedy(G, t_set, t_nodes_list, 1)

        for num_of_t_nodes, t_nodes_set in params_dict.items():
            num_of_commits, num_of_rounds = broadcast(G, faulty_nodes=set(), trusted_nodes = t_nodes_set)
            result_dict[num_of_t_nodes][7].append(num_of_rounds)

    # iterate the default dict to generate a normal dict for serializing
    outside_dict = dict()
    for k, v in result_dict.items():
        inside_dict = dict()
        for k2, v2 in v.items():
            inside_dict[k2] = v2
        outside_dict[k] = inside_dict
    return outside_dict

def broadcast(graph, faulty_nodes=set(), trusted_nodes=set()):
    """
    Given a potentially desired graph and a set of faulty nodes,
    do a broadcasting to see whether all non-faulty nodes are able to receive the source's commit
    :param graph: a graph generated by propose_graph()
    :param faulty_nodes: a set of faulty nodes picked by doing combination
    :param trusted_nodes:
    :return: the number of non-faulty nodes that have committed the message, including the source
    """

    # Round 0: initialize and the source commits
    curr_round = 0
    # every non-faulty node (except the src) appears in non_faulty_commit_queue only once,
    # but all faulty nodes will not be in this queue
    non_faulty_commit_queue = deque()
    non_faulty_commit_queue.append(0)

    # to record whether a node has committed the value before (due to cyclic graph),
    # we put the node to the set if it is the source, or
    # it receives a commit from the source/trusted node, or
    # it receives (MAX_FAULT_NODES + 1) commits from incoming nodes (faulty nodes don't commit)
    non_faulty_has_committed = {0}

    # to record the number of proposes a node receives if it's not directly linked to the source
    propose_received = defaultdict(lambda: 0)

    # Round >= 1: all non-faulty nodes commits
    while len(non_faulty_commit_queue):  # while not all nodes have committed
        curr_round += 1
        for curr_node in range(len(non_faulty_commit_queue)):  # for all nodes in the current round of commits
            curr_node = non_faulty_commit_queue.popleft()
            curr_node_neis = [edge[1] for edge in graph.edges(curr_node)]  # all outgoing neighbours of the current node

            if curr_node in trusted_nodes:  # if this commit comes from the source or trusted nodes
                for nei in curr_node_neis:
                    # If this node has committed before (due to cyclic graph) OR if this node is faulty, ignore it;
                    if nei in non_faulty_has_committed or nei in faulty_nodes:
                        continue

                    non_faulty_commit_queue.append(nei)
                    non_faulty_has_committed.add(nei)
            else:
                for nei in curr_node_neis:
                    # If this node has committed before (due to cyclic graph) OR if this node is faulty, ignore it;
                    if nei in non_faulty_has_committed or nei in faulty_nodes:
                        continue

                    # If this node is non-faulty, it commits iff it has heard (MAX_FAULT_NODES + 1) non-faulty proposes.
                    # note: faulty nodes don't propose values
                    # TODO: does MAX_FAULTY_NODES logic goes well with giant graph? yes?
                    propose_received[nei] += 1
                    if propose_received[nei] >= MAX_NUMBER_OF_FAULT_NODES + 1:
                        non_faulty_commit_queue.append(nei)
                        non_faulty_has_committed.add(nei)
    return len(non_faulty_has_committed), curr_round


if __name__ == '__main__':
    pass

    G = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/subgraphs/geo_300/geo_300_0.15_1", "rb"))
    pick_trusted_centrality(G, 10, [0], 3, 1)
    # result_dict = simulation_batch()
    # pickle.dump(result_dict, open("remove_greedy_geo_1.p", "wb"))

    # pickle.dump(open("remove_greedy_geo_1.p", "rb"))
    # print(result_dict)
