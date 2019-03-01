import networkx as nx
from collections import deque
from random import *
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
from time import time
MAX_NUMBER_OF_FAULT_NODES = 3


# Check if the vertex is a faulty node
def is_fault_node(vertex, fault_nodes):
    return vertex in fault_nodes


# Generate the graph we want to test on
def test_graph():
    # 100_17_100_0.3_200_0.2_300_0.17_300_0.2
    trust_nodes = {100, 200, 400, 700}
    bad_nodes = [82, 70, 28, 105, 158, 178, 322, 364, 389, 499, 660, 636, 710, 811, 823]
    graph = pickle.load(open("100_17_100_0.3_200_0.2_300_0.17_300_0.2.p", "rb"))
    return bad_nodes, trust_nodes, graph


# Do the broadcast algorithm and also draw out the graph
def broadcast_main(fault_list, trusted_nodes, graph):
    print("Graph_Edges:", graph.edges)
    round_dict = broadcast_entry(graph, fault_list, trusted_nodes)
    print("Number of edges", len(list(graph.edges())))
    # Draw the graph
    from Gui import Sim
    Sim(g=graph, d=round_dict)


# By default we will randomly choose bad set and no trusted Node
def broadcast_entry(graph, fault_list, trusted_nodes):
    # Initialize variables needed in broad_cast function
    count = dict()
    commit = [False] * (len(graph.nodes))
    number_of_fault_nodes = MAX_NUMBER_OF_FAULT_NODES
    round_dict = dict()
    for i in range((len(graph.nodes))):
        count[i] = dict()
    return broadcast(count, commit, number_of_fault_nodes, graph, round_dict, fault_list, trusted_nodes)


def broadcast(count: dict, commit: list, number_of_fault_nodes: int, graph, round_dict: dict, fault_nodes, trusted_nodes):
    print("Bad:", fault_nodes)
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

        # This two list will track each round, which good and bad nodes commit
        # For drawing and debugging purpose
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
                print(nei)
                if commit[nei]:
                    continue

                # If current Node is src then push this node into the Queue
                # because all the outgoing neighbors for the src node will directly commit the value
                elif current_node == 0 or (trusted_nodes is not None and current_node in trusted_nodes):
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
    # print(count)
    return round_dict


def main():
    bad_nodes, trusted_nodes, graph = test_graph()
    broadcast_main(bad_nodes, trusted_nodes, graph)


if __name__ == '__main__':
    main()