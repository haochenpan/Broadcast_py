import networkx as nx
from collections import deque
from random import *
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
from time import time

TOTAL_NODES = 400
MAX_NUMBER_OF_FAULT_NODES = 3

# Total edge = EDGE_FACTOR * TOTAL_NODES
EDGE_FACTOR = 18

THRESHOLD = 0.15


# Parameter num_Edges will only be used when doing binomial graph
# For geometric, it will only use Threshold
def buildGraph(num_edges: int):
    # graph = nx.gnm_random_graph(TOTAL_NODES, num_edges)
    graph = nx.random_geometric_graph(TOTAL_NODES, THRESHOLD)
    # nx.draw(graph)
    # plt.show()
    return graph


# Restricting the number of nodes around the src to simulate kind of a worst case situation (most f + 1 cases)
def restrict_src_good_neis(graph, fault_nodes):
    src_edges = list(graph.edges(0))
    # print("Src_edges", src_edges)

    num_good_src_near_neighbor = MAX_NUMBER_OF_FAULT_NODES * 2 + 1
    # Restricting the number of neis around the src so that we can see more f + 1 condition
    # Also notice that src node needs to have at least 2 * f + 1 neis around it
    if len(graph.edges) < num_good_src_near_neighbor:
        return False

    list_src_good_neis = []

    for i in range(len(src_edges)):
        node = src_edges[i][1]
        # If it is a bad node, we do not need to restrict it because we are only restricting good neis around src
        if not is_fault_node(node, fault_nodes):
            list_src_good_neis.append(src_edges[i][1])

    # Restricting the good neis around src
    while len(list_src_good_neis) > num_good_src_near_neighbor:
        removed_node = list_src_good_neis[len(list_src_good_neis) - 1]
        list_src_good_neis.remove(removed_node)
        graph.remove_edge(0, removed_node)

    return True


# Check all possible sets of the fault nodes to check overall validity
def check_valid(graph):
    # Early termination for check_Valid if the graph is not in the single connected component
    if nx.number_connected_components(graph) > 1:
        print("not 1 component")
        return False

    for num_fault in range(MAX_NUMBER_OF_FAULT_NODES + 1):
        nodes_id_list = list(graph.nodes)
        # Node 0 is always the src node
        nodes_id_list.remove(0)
        # Return all the possible combination list (n chooses num_falut)
        combination_fault_nodes_list = combination_pick(nodes_id_list, num_fault)

        for bad_nodes_instance in combination_fault_nodes_list:
            count = dict()
            commit = [False] * TOTAL_NODES
            round_dict = dict()
            fault_nodes = set(bad_nodes_instance)
            # print("Bad List: ", fault_nodes)
            # Meaning the src nei does not have 2 * f + 1 node
            if not restrict_src_good_neis(graph, fault_nodes):
                return False

            for i in range(len(graph.node)):
                # print(graph.edges(i))
                count[i] = dict()

            if not validGraph(count, commit, num_fault, graph, round_dict, fault_nodes):
                print ("This graph fails")
                return

    print("This graph Succeed")
    return graph


# This function will generate all possible combination of faulty nodes
def combination_pick(id_list, number_of_fault_nodes):
    return combinations(id_list, number_of_fault_nodes)


# Check if the vertex is a faulty node
def is_fault_node(vertex, fault_nodes):
    return vertex in fault_nodes


def instance_succeed(result_count, number_of_fault_nodes):
    return result_count == TOTAL_NODES - number_of_fault_nodes


# We put a node into the Queue iff
# 1. it is able to commit the value (src value)
# 2. It is a fault_node
def validGraph(count: dict, commit: list, number_of_fault_nodes: int, graph, round_dict: dict, fault_nodes):
    # Count how many good nodes commit to the value
    good_commit_count = 0
    rounds = 0

    # Queue data Structure for doing breadth first search
    # Always start with the source
    q = deque()
    q.append(0)
    commit[0] = True


    # Src commit
    round_dict[rounds] = [[0], []]
    rounds += 1
    good_commit_count += 1

    # Synchronized Network Broadcast using the idea of breadth first search
    # Since each round there is at least one node that is going to commit the value
    # So this while loop ends iff all the nodes in the graph have commited the value
    while not len(q) == 0:
        cur_size = len(q)

        bad_node_commit = []
        good_node_commit = []

        for i in range(cur_size):
            # Get the out_going_neighbors of current node
            # A list of tuple of outgoing edges
            current_node = q.popleft()
            list_of_edges = graph.edges(current_node)

            for out_edge in list_of_edges:
                nei = out_edge[1]

                # If this node is commited already, then ignore it.
                if commit[nei]:
                    continue

                # If current Node is src then push this node into the Queue
                # because all the outgoing neighbors for the src node will directly commit the value
                elif current_node == 0:
                    q.append(nei)
                    commit[nei] = True
                    if not is_fault_node(nei,fault_nodes):
                        # print("ID:" + str(nei) + " has information " + str(0) + " from direct neighbor", str(rounds))
                        good_node_commit.append(nei)
                        good_commit_count += 1
                    else:
                        bad_node_commit.append(nei)

                # A fault_free_node but only commit and broadcast the value
                # iff it matches the condition of at least f + 1
                else:
                    # Whenever we reach a fault Node, it is still able to propagate, so we push it into the queue
                    if is_fault_node(nei,fault_nodes):
                        q.append(nei)
                        commit[nei] = True
                        bad_node_commit.append(nei)
                    # We node that FAULT_FREE_NODES that have commited have the value of the src node
                    else:
                        if not is_fault_node(current_node, fault_nodes):
                            if 0 not in count[nei].keys():
                                count[nei][0] = 1
                            else:
                                count[nei][0] += 1
                                # If reach the f + 1 condition, can commit and broadcast

                            if count[nei][0] >= number_of_fault_nodes + 1:
                                q.append(nei)
                                commit[nei] = True
                                # print("ID:" + str(nei) + " has information " + str(0) +
                                #       " from f + 1 condition with count " + str(count[nei][0]), str(rounds))
                                good_node_commit.append(nei)
                                good_commit_count += 1

                        # This part can actually be taken away
                        # since we already know that fault nodes will never be commited
                        # But still write it out for a clearer picture
                        # Fault nodes simply broadcast their id to their outgoing neighbors
                        # else:
                        #     if current_node not in count[nei].keys():
                        #         count[nei][current_node] = 1
                        #     else:
                        #         count[nei][current_node] += 1

        # round_dict[rounds] = [good_node_commit, bad_node_commit]
        # print("Good node commit in round" + str(rounds), good_node_commit)
        # print("Bad node commit in round" + str(rounds), bad_node_commit)
        # print(len(q))
        rounds += 1

    return instance_succeed(good_commit_count, number_of_fault_nodes)


def broadcast_entry(graph):
    count = dict()
    commit = [False] * TOTAL_NODES
    number_of_fault_nodes = MAX_NUMBER_OF_FAULT_NODES
    round_dict = dict()
    for i in range (TOTAL_NODES):
        count[i] = dict()
    nodes_id_list = list(graph.nodes)
    # Node 0 is always the src node
    nodes_id_list.remove(0)

    combination_fault_list = list(combination_pick(nodes_id_list, number_of_fault_nodes))
    pick_index = randint(0, len(combination_fault_list) - 1)
    fault_nodes_list = combination_fault_list[pick_index]
    # print("Fault nodes are", list(fault_nodes_list))
    return broadcast(count, commit, number_of_fault_nodes, graph, round_dict, fault_nodes_list)


def broadcast(count: dict, commit: list, number_of_fault_nodes: int, graph, round_dict: dict, fault_nodes):
    # Count how many good nodes commit to the value
    good_commit_count = 0
    rounds = 0

    # Queue data Structure for doing breadth first search
    # Always start with the source
    q = deque()
    q.append(0)
    commit[0] = True

    # Src commit
    round_dict[rounds] = [[0], []]
    rounds += 1
    good_commit_count += 1

    # Synchronized Network Broadcast using the idea of breadth first search
    # Since each round there is at least one node that is going to commit the value
    # So this while loop ends iff all the nodes in the graph have commited the value
    while not len(q) == 0:
        cur_size = len(q)

        bad_node_commit = []
        good_node_commit = []

        for i in range(cur_size):
            # Get the out_going_neighbors of current node
            # A list of tuple of outgoing edges
            current_node = q.popleft()
            list_of_edges = graph.edges(current_node)

            for out_edge in list_of_edges:
                nei = out_edge[1]

                # If this node is commited already, then ignore it.
                if commit[nei]:
                    continue

                # If current Node is src then push this node into the Queue
                # because all the outgoing neighbors for the src node will directly commit the value
                elif current_node == 0:
                    q.append(nei)
                    commit[nei] = True
                    if not is_fault_node(nei,fault_nodes):
                        # print("ID:" + str(nei) + " has information " + str(0) + " from direct neighbor", str(rounds))
                        good_node_commit.append(nei)
                        good_commit_count += 1
                    else:
                        bad_node_commit.append(nei)

                # A fault_free_node but only commit and broadcast the value
                # iff it matches the condition of at least f + 1
                else:
                    # Whenever we reach a fault Node, it is still able to propagate, so we push it into the queue
                    if is_fault_node(nei,fault_nodes):
                        q.append(nei)
                        commit[nei] = True
                        bad_node_commit.append(nei)
                    # We node that FAULT_FREE_NODES that have commited have the value of the src node
                    else:
                        if not is_fault_node(current_node, fault_nodes):
                            if 0 not in count[nei].keys():
                                count[nei][0] = 1
                            else:
                                count[nei][0] += 1
                                # If reach the f + 1 condition, can commit and broadcast

                            if count[nei][0] >= number_of_fault_nodes + 1:
                                q.append(nei)
                                commit[nei] = True
                                # print("ID:" + str(nei) + " has information " + str(0) +
                                #       " from f + 1 condition with count " + str(count[nei][0]), str(rounds))
                                good_node_commit.append(nei)
                                good_commit_count += 1

                        # This part can actually be taken away
                        # since we already know that fault nodes will never be commited
                        # But still write it out for a clearer picture
                        # Fault nodes simply broadcast their id to their outgoing neighbors
                        # else:
                        #     if current_node not in count[nei].keys():
                        #         count[nei][current_node] = 1
                        #     else:
                        #         count[nei][current_node] += 1

        round_dict[rounds] = [good_node_commit, bad_node_commit]
        print("Good node commit in round" + str(rounds), good_node_commit)
        print("Bad node commit in round" + str(rounds), bad_node_commit)
        # print(len(q))
        rounds += 1
    print(count)
    return round_dict


def test_graph_main():
    num_edges = TOTAL_NODES * EDGE_FACTOR
    while True:
        print("********************************************************************")
        print(num_edges)
        graph = buildGraph(num_edges)
        if check_valid(graph):
            break
    print("Save to pickels")
    pickle.dump(graph, open(f"{TOTAL_NODES}_{THRESHOLD}node_bin_1{int(time())}.p", "wb"))


def broadcast_main():
    graph = pickle.load(open("100_10node_bin_11551281716.p", "rb"))
    print(str(len(graph.edges)))
    print("Graph_Edges:",graph.edges)
    round_dict = broadcast_entry(graph)
    print("Number of edges", len(list(graph.edges())))
    from Gui import Sim
    Sim(g=graph, d=round_dict)


# Check a single graph validity for debugging purpose
def check_current_graph_valid():
    graph = pickle.load(open("200_0", "rb"))
    print(check_valid(graph))
    nx.draw(graph)
    plt.show()


def run_five_times():
    for i in range (5):
        test_graph_main()


def main():
    # test_graph_main()
    broadcast_main()




if __name__ == '__main__':
    main()