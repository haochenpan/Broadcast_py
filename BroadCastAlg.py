import networkx as nx
from collections import deque
from random import *
import matplotlib.pyplot as plt

NUMBER_OF_FAULT_FREE_NODES = 10
NUMBER_OF_FAULT_NODES = 3
TOTAL_NODES = NUMBER_OF_FAULT_FREE_NODES + NUMBER_OF_FAULT_NODES
# Adding some control over the graph
NUMBER_OF_GOOD_NODES_AROUND_SRC = 1

# Least number of outgoing edges
# Have some control over number of outgoing edges to make our graph easier to view
# This number needs to be at least 1 because we do not want more than 1 connected component
NUMBER_LEAST_OUT_GOING = 1

# List that will be helpful for creating the graph
HELPER_LIST = list(range(TOTAL_NODES))
GOOD_NODES = HELPER_LIST[0:NUMBER_OF_FAULT_FREE_NODES]
BAD_NODES = HELPER_LIST[NUMBER_OF_FAULT_FREE_NODES:TOTAL_NODES]

# Key:node's id
# value list of list:
# first sublist: good neis commit, second sublist: bad neis
# Look like the following
# 0: [0][] Since first round only src commit
# 1: [0's good nei][0's bad_nei]
# ...
round_dict = dict()


# Testing that only one good node around src
def connect_incoming_edges(graph, vertex):
    # No incoming edges for vertex
    if vertex == 0:
        return

    local_list = HELPER_LIST[:]
    # Choices for the neighbors
    local_bad_neis = BAD_NODES[:]

    # We will handle incoming edges from src in connect_outgoing_edges
    local_good_neis = GOOD_NODES[1:]

    # if bad nodes, the incoming does not need to have any constraint
    if vertex >= NUMBER_OF_FAULT_FREE_NODES:
        # No self-loop in this graph
        local_list.remove(vertex)
        # Randomly pick the number of incoming neighbors

        num_of_incoming_neis = randint(1, len(local_list))
        # Randomly pick nodes to be the neighbors
        shuffle(local_list)
        for i in range(len(local_list)):
            graph.add_edge(local_list[i], vertex)
        return

    # If vertex is a good node, the incoming nodes need to have some constraints
    # in which the vertex has to have at least f + 1 incoming good node

    # Randomly pick the number of Fault_Nodes
    num_of_bad_neis = randint(0, len(local_bad_neis))
    shuffle(local_bad_neis)
    for i in range(num_of_bad_neis):
        graph.add_edge(local_bad_neis[i], vertex)

    # Remove the self-loop edge
    local_good_neis.remove(vertex)
    # Randomly pick the number of Fault_Free_Nodes around current vertex
    # Needs to have at least N+1 good nodes around
    num_of_good_neis = randint(NUMBER_OF_FAULT_NODES + 1, len(local_good_neis))
    shuffle(local_good_neis)
    for i in range(num_of_good_neis):
        graph.add_edge(local_good_neis[i], vertex)


def connect_outgoing_edges(graph, vertex):
    local_list = HELPER_LIST[:]
    local_bad_neis = BAD_NODES[:]
    local_good_neis = GOOD_NODES[:]

    # Need to prevent the case that src does not have any good_out_neis
    if vertex == 0:
        local_good_neis.remove(0)
        shuffle(local_bad_neis)
        shuffle(local_good_neis)
        num_of_bad_neis = randint(0, len(local_bad_neis))
        for i in range(num_of_bad_neis):
            graph.add_edge(vertex, local_bad_neis[i])

        num_of_good_neis = NUMBER_OF_GOOD_NODES_AROUND_SRC
        for i in range(num_of_good_neis):
            graph.add_edge(vertex, local_good_neis[i])

    # We still need to do this to prevent a node in its own component
    else:
        # prevent self-loop
        num_out_going = len((graph.out_edges(vertex)))
        if num_out_going < NUMBER_LEAST_OUT_GOING:
            local_list.remove(vertex)
            # Prevent one node stay in its own component, also add some control over the outgoing edges
            num_out_going_neis = NUMBER_LEAST_OUT_GOING - num_out_going
            for i in range (num_out_going_neis):
                graph.add_edge(vertex, local_list[i])


def is_good_node(vertex):
    return vertex < NUMBER_OF_FAULT_FREE_NODES


def buildgraph():
    # index represents individual nodes id ... only need to keep track the id in this case
    # Corresponding value is another map
    # the inner map's key is the information and the value represents the number of this information this node gets

    graph = nx.DiGraph()
    # Key: id of nodes
    # Value: Also a dict in which its key is the information received and value is the count of this information
    # Useful for f + 1 condition
    count = dict()
    # Like a visited array if bfs that prevents revisiting
    commit = [False] * TOTAL_NODES

    # create Nodes and Edges in the graph
    for i in range(TOTAL_NODES):
        # Also create a dictionary inside the count
        count[i] = dict()
        # This function will build the vertex and all its incoming neighbors
        connect_incoming_edges(graph, i)
        connect_outgoing_edges(graph, i)

    # Plot the initial graph here
    for i in range (TOTAL_NODES):
        print(graph.in_edges(i))
    # nx.draw(graph, with_labels= True)
    # plt.show()
    return count, commit, graph


# We put a node into the Queue iff
# 1. it is able to commit the value (src value)
# 2. It is a fault_node
def broadcast(count: dict, commit: list, graph):
    # Queue data Structure for doing breadth first search
    # Always start with the source
    q = deque()
    q.append(0)
    commit[0] = True

    rounds = 0
    # Src commit
    round_dict[rounds] = [[0], []]
    rounds += 1

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
            neighbors = graph.out_edges(current_node)

            for out_tuple in neighbors:
                nei = out_tuple[1]

                # If this node is commited already, then ignore it.
                if commit[nei]:
                    continue

                # If current Node is src then push this node into the Queue
                # because all the outgoing neighbors for the src node will directly commit the value
                elif current_node == 0:
                    q.append(nei)
                    commit[nei] = True
                    if is_good_node(nei):
                        print("ID:" + str(nei) + " has information " + str(0) + " from direct neighbor", str(rounds))
                        good_node_commit.append(nei)
                    else:
                        bad_node_commit.append(nei)

                # Whenever we reach a fault Node, it is still able to propagate, so we push it into the queue
                elif not is_good_node(nei):
                    q.append(nei)
                    commit[nei] = True
                    bad_node_commit.append(nei)

                # A fault_free_node but only commit and broadcast the value
                # iff it matches the condition of at least f + 1
                else:
                    # We node that FAULT_FREE_NODES that have commited have the value of the src node
                    if current_node < NUMBER_OF_FAULT_FREE_NODES:
                        if 0 not in count[nei].keys():
                            count[nei][0] = 1
                        else:
                            count[nei][0] += 1
                            # If reach the f + 1 condition, can commit and broadcast
                            if count[nei][0] >= NUMBER_OF_FAULT_NODES + 1:
                                q.append(nei)
                                commit[nei] = True
                                print("ID:" + str(nei) + " has information " + str(0) +
                                      " from f + 1 condition with count " + str(count[nei][0]), str(rounds))
                                good_node_commit.append(nei)

                    # This part can actually be taken away since we already know that fault nodes will never be commited
                    # But still write it out for a clearer picture
                    # Fault nodes simply broadcast their id to their outgoing neighbors
                    else:
                        if current_node not in count[nei].keys():
                            count[nei][current_node] = 1
                        else:
                            count[nei][current_node] += 1

        round_dict[rounds] = [good_node_commit, bad_node_commit]
        rounds += 1


def main():
    count, commit, graph = buildgraph()
    broadcast(count, commit, graph)
    print(round_dict)


if __name__ == '__main__':
    main()