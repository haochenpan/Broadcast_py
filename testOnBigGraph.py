import pickle
import networkx as nx
from collections import defaultdict, OrderedDict, deque
from operator import itemgetter
import os
from random import shuffle
from random import sample
from itertools import combinations
import numpy as np


MAX_NUMBER_OF_FAULT_NODES = 3


def broadcast(graph, trusted_nodes=set(), faulty_nodes=set()):
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


# Pick the top k among the trusted
def centrality_top_k(centrality_dict, num_of_trusted, t_node_set):
    good_nodes_list = []

    count = 0
    for k, v in centrality_dict.items():
        count += 1
        good_nodes_list.append(k)
        if count == num_of_trusted:
            break
    return good_nodes_list


# 1 Closeness           4: Rmove with Closeness
# 2 Betweeness          5. Remove with betweeness
# 3. Degree             6. degreeCentrality
def pick_trusted_centrality(G, num_trusted, t_node_set, alg_num, overlap):
    if num_trusted == 0:
        return set()

    if alg_num == 1 or alg_num == 4:
        centrality_dict = nx.closeness_centrality(G)
    elif alg_num == 2 or alg_num == 5:
        centrality_dict = nx.betweenness_centrality(G)
    elif alg_num == 3 or alg_num == 6:
        centrality_dict = nx.degree_centrality(G)

    centrality_dict = remove_src(centrality_dict, t_node_set)

    centrality_dict = order_dict(centrality_dict)

    if alg_num == 1 or alg_num == 2 or alg_num == 3:
        return centrality_top_k(centrality_dict, num_trusted, t_node_set)

    elif alg_num == 4 or alg_num == 5 or alg_num == 6:
        return get_trusted(G, centrality_dict, num_trusted, overlap)


# Remove the src from being a candidate for trusted node
def remove_src(centrality_dict, t_node_set):
    for src_id in t_node_set:
        del centrality_dict[src_id]
    return centrality_dict


def get_trusted(G, centrality_dict: dict, num_trusted: int, overlap: int):
    return_trusted_nodes = list()

    while num_trusted > 0:
        copy_dict = order_dict(dict(centrality_dict))

        centrality_dict = dict()

        for id in copy_dict.keys():
            # Check if this id can be picked as trusted based on threshold
            if checkTrusted(G, id, return_trusted_nodes, overlap):
                return_trusted_nodes.append(id)
                num_trusted -= 1
                if num_trusted == 0:
                    return return_trusted_nodes

        for node, value in copy_dict.items():
            if node not in return_trusted_nodes:
                centrality_dict[node] = value
        overlap = overlap + 1

    return return_trusted_nodes


# Check if the current node has overlap more than the threshold
def checkTrusted(G, id, return_trusted_nodes, overlap):
    return_trusted_nodes = set(return_trusted_nodes)
    for trusted_id in return_trusted_nodes:
        common_neis = len(list(nx.common_neighbors(G, id, trusted_id)))
        if common_neis > overlap:
            return False
    return True


def order_dict(centrality_dict):
    centrality_dict = OrderedDict(sorted(centrality_dict.items(), key=itemgetter(1), reverse=True))
    return centrality_dict


def loadGraph(pickle_name) :
    G = pickle.load(open(pickle_name, "rb"))
    return G

#Key:
    # (Graph_ID_
# Key: alg_num
# Value: list of good nodes picked


def similation():
    t_node_set = {0, 300, 600, 900, 1200}
    num_trusted_list = list(range(0, 300, 10))

    # {key: num of trusted; value: {key: algo enum; value: num of rounds}}
    result_dict = dict()

    for num_trusted in num_trusted_list:
        result_dict[num_trusted] = dict()
        for i in range(1, 7):
            result_dict[num_trusted][i] = []

    # Put all those tested graph into the list
    tested_graphs_name = []

    for i in range(12):
        tested_graphs_name.append("Big_Graph_" + str(i))

    graph_index = 0

    G = loadGraph(tested_graphs_name[graph_index])

    alg_num = 1
    # 7 algorithms
    while alg_num < 7:
        print("Alg_num",alg_num)
        for num_trusted in num_trusted_list:
            print("Num_Trusted", num_trusted)
            current_trusted = pick_trusted_centrality(G, num_trusted, t_node_set, alg_num, 3)
            current_trusted.update(t_node_set.copy())
            _, rounds = broadcast(G, current_trusted)

            result_dict[num_trusted][alg_num].append(rounds)
        alg_num += 1

    print(result_dict)

    pickle.dump(result_dict, open(f"Big_Graph_{graph_index}.p", "wb"))
    return result_dict


def simulation_good_nodes_picked():
    t_node_set = {0, 300, 600, 900, 1200}
    num_trusted_list = list(range(0, 300, 10))

    # {key: num of trusted; value: {key: algo enum; value: num of rounds}}
    result_dict = dict()

    # Put all those tested graph into the list
    tested_graphs_name = []

    for i in range(12):
        tested_graphs_name.append("Big_Graph_" + str(i))

    # Key: graph_id
    # value: small dict:
    # key: alg_name
    # Top 70 trusted nodes
    good_nodes_dict = dict()
    for i in range (12):
        good_nodes_dict[i] = dict()
        for j in range(1, 7):
            good_nodes_dict[i][j] = list()

    num_trusted = 150
    for graph_index in range(12):
        G = loadGraph(tested_graphs_name[graph_index])
        alg_num = 1
        # 7 algorithms
        print(graph_index)
        while alg_num < 7:
            print("Alg_num", alg_num)
            current_trusted = pick_trusted_centrality(G, num_trusted, t_node_set, alg_num, 3)
            current_trusted.extend(t_node_set.copy())
            good_nodes_dict[graph_index][alg_num] = current_trusted
            alg_num += 1

    pickle.dump(good_nodes_dict, open(f"Big_Graph_trusted.p", "wb"))
    return result_dict


def simulation_good_nodes_picked_TOP50():
    result_dict = pickle.load(open("Big_Graph_trusted.p", "rb"))
    for graph_index, small_dict in result_dict.items():
        for alg_num, list in small_dict.items():
            result_dict[graph_index][alg_num] = list[0:50]

    pickle.dump(result_dict, open(f"Big_Graph_trusted_Top50.p", "wb"))


def simulation_load_TOP50():
    result_dict = pickle.load(open("Big_Graph_trusted_TOP50.p", "rb"))
    # for graph_index, small_dict in result_dict.items():
    #     print('Graph_Index', graph_index)
    #     for alg_num, lst in small_dict.items():
    #         print(alg_num, lst)
    return result_dict


def simulation_load_Overall():
    result_dict = pickle.load(open("Big_Graph_trusted.p", "rb"))
    return result_dict


def write_to_txt(file_Name):
    G = pickle.load(open(file_Name, "rb"))

    file = open(file_Name+"txt","w")

    for i in range(1500):
        edge = nx.edges(G, i)
        edge = list(edge)
        file.write(str(i) + "\n")
        for item in edge:
            file.write("%s\n" % str(item))
    file.close()


# def checkGoodNodeInfo():
#     result_dict = pickle.load(open("Big_Graph_trusted.p", "rb"))
#     for graph_id, small_dict in result_dict.items():
#         # print("Graph_ID", graph_id)
#         for alg_num, trusted in small_dict.items():
#             # print (alg_num, trusted)


def checklinks():
    # Put all those tested graph into the list
    tested_graphs_name = []
    result_dict = dict()
    for i in range(12):
        result_dict[i] = set()

    for j in range(12):
        tested_graphs_name.append("Big_Graph_" + str(j))
        G = loadGraph(tested_graphs_name[j])
        for i in range(1500):
            edges = nx.edges(G, i)
            for t in edges:
                nei = t[1]
                if (t[1] - i >= 300):
                    result_dict[j].add(i)

    return result_dict


def check_overlap():
    # key: graph_id
    # value: small dict:
            #key: alg_num
            #value: top60 trusted nodes
    # result_dict = simulation_good_nodes_picked()
    result_dict = simulation_load_TOP50()
    # key: graph_id
    # value: set src_links
    link_dict = checklinks()

    count_dict = dict()
    for i in range(12):
        count_dict[i] = dict()


    for graph_id, small_dict in result_dict.items():
        for alg_num, trusted in small_dict.items():
            count = 0
            print(trusted)
            for nodes in trusted:
                if nodes in link_dict[graph_id]:
                    count += 1
            count_dict[graph_id][alg_num] = count

    return count_dict


def main():
    count_dict = check_overlap()
    for graph_id, small_dict in count_dict.items():
        print("Graph_Number", graph_id)
        for alg_num, count in small_dict.items():
            if alg_num == 1:
                alg_name = "Closeness"
            elif alg_num == 2:
                alg_name = "Betweeness"
            elif alg_num == 3:
                alg_name = "Degree"
            elif alg_num == 4:
                alg_name = "Closeness + Remove"
            elif alg_num == 5:
                alg_name = "Betweeness + Remove"
            elif alg_num == 6:
                alg_name = "Degree + Remove"

            print(alg_name, count)


    # for num_trusted, small_dict in result_dict.items():
    #     print ("Number of trusted", num_trusted)
    #     for alg_num, rounds in small_dict.items():
    #         print (alg_num, rounds)
    # result_dict = pickle.load(open("/Users/yingjianwu/Desktop/Big_Graph_0.p","rb"))



if __name__ == '__main__':
    pass
    main()