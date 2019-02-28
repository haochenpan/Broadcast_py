import networkx as nx
from collections import deque
from random import *
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
from time import time

TOTAL_NODES = 400
MAX_NUMBER_OF_FAULT_NODES = 3

# Used for randomGraph
# Total edge = EDGE_FACTOR * TOTAL_NODES
EDGE_FACTOR = 18

# Used for geometricGraph
THRESHOLD = 0.15


##############################
# Build Graph Model Part
##############################

# Parameter num_Edges will only be used when doing random graph
def build_random_graph():
    num_edges = TOTAL_NODES * EDGE_FACTOR
    graph = nx.gnm_random_graph(TOTAL_NODES, num_edges)
    return graph


# For geometric, it will only use Threshold
def build_random_geo_graph():
    graph = nx.random_geometric_graph(TOTAL_NODES, THRESHOLD)
    return graph


##############################
# Validty Part
##############################

# Restricting the number of nodes around the src to simulate kind of a worst case situation (most f + 1 cases)
def restrict_src_good_neis(graph, fault_nodes):
    # First get all edges from src
    src_edges = list(graph.edges(0))

    # Restricting the number of neis around the src so that we can see more f + 1 condition
    # Need to at least statisfies this condition to make the totality of validity feasible
    num_good_src_near_neighbor = MAX_NUMBER_OF_FAULT_NODES * 2 + 1

    # Early termination condition
    # Also notice that src node needs to have at least 2 * f + 1 neis around it
    # If not this graph will definitely not work

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


# Check all possible sets of the fault nodes to check overall validity
def check_valid(graph):
    # Early termination for check_Valid if the graph is not in the single connected component
    if nx.number_connected_components(graph) > 1:
        print("not 1 component")
        return False

    # If src does not have enough neis, directly return false
    # Early Termination
    if len(graph.edges(0)) < MAX_NUMBER_OF_FAULT_NODES * 2 + 1:
        return False

    for num_fault in range(MAX_NUMBER_OF_FAULT_NODES + 1):
        nodes_id_list = list(graph.nodes)
        # Node 0 is always the src node
        nodes_id_list.remove(0)
        # Return all the possible combination list (n chooses num_falut)
        combination_fault_nodes_list = combination_pick(nodes_id_list, num_fault)

        for bad_nodes_instance in combination_fault_nodes_list:
            count = dict()
            commit = [False] * (len(graph.nodes))
            # round_dict = dict()
            fault_nodes = set(bad_nodes_instance)
            # print("Bad List: ", fault_nodes)

            # Meaning the src nei does not have 2 * f + 1 node
            restrict_src_good_neis(graph, fault_nodes)

            for i in range(len(graph.nodes)):
                # print(graph.edges(i))
                count[i] = dict()

            if not validGraph(count, commit, num_fault, graph, fault_nodes):
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


# Result_count_will be the total count of the good node after running the checking graph
def instance_succeed(result_count, number_of_fault_nodes, graph):
    return result_count == len(graph.nodes) - number_of_fault_nodes


# We put a node into the Queue iff
# 1. it is able to commit the value (src value)
# 2. It is a fault_node
# 3. Count is used to count the number of information get => f + 1 condition
# 4. Commit is like a visited Array
# 5. fault_nodes: a list of fault nodes for now
def validGraph(count: dict, commit: list, number_of_fault_nodes: int, graph, fault_nodes):
    # Count how many good nodes commit to the value
    good_commit_count = 0
    rounds = 0

    # Queue data Structure for doing breadth first search
    # Always start with the source
    q = deque()
    q.append(0)
    commit[0] = True


    # Src commit
    # round_dict[rounds] = [[0], []]
    rounds += 1
    good_commit_count += 1

    # print(fault_nodes)
    # Synchronized Network Broadcast using the idea of breadth first search
    # Since each round there is at least one node that is going to commit the value
    # So this while loop ends iff all the nodes in the graph have commited the value
    while not len(q) == 0:
        cur_size = len(q)

        # bad_node_commit = []
        # good_node_commit = []

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
                    if not is_fault_node(nei, fault_nodes):
                        # print("ID:" + str(nei) + " has information " + str(0) + " from direct neighbor", str(rounds))
                        # good_node_commit.append(nei)
                        good_commit_count += 1
                    # else:
                        # bad_node_commit.append(nei)

                # A fault_free_node but only commit and broadcast the value
                # iff it matches the condition of at least f + 1
                else:
                    # Whenever we reach a fault Node, it is still able to propagate, so we push it into the queue
                    if is_fault_node(nei, fault_nodes):
                        q.append(nei)
                        commit[nei] = True
                        # bad_node_commit.append(nei)
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
                                good_commit_count += 1
                                # print("ID:" + str(nei) + " has information " + str(0) +
                                #       " from f + 1 condition with count " + str(count[nei][0]), str(rounds))
                                # good_node_commit.append(nei)

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
    # print(good_commit_count)
    return instance_succeed(good_commit_count, number_of_fault_nodes, graph)

##############################
# Broadcast Part
##############################


# By default we will randomly choose bad set and no trusted Node
def broadcast_entry(graph, fault_list = None, trusted_nodes= None):
    # Initialize variables needed in broad_cast function
    count = dict()
    commit = [False] * (len(graph.nodes))
    number_of_fault_nodes = MAX_NUMBER_OF_FAULT_NODES
    round_dict = dict()
    for i in range((len(graph.nodes))):
        count[i] = dict()
    nodes_id_list = list(graph.nodes)
    # Node 0 is always the src node
    nodes_id_list.remove(0)

    # Here we can have choices: if you know the bad nodes you want to target at
    # Then can get rid of these three lines and passed in the specific bad nodes you want to test
    # This is basically getting a random set of bad nodes
    if fault_list is None:
        combination_fault_list = list(combination_pick(nodes_id_list, number_of_fault_nodes))
        pick_index = randint(0, len(combination_fault_list) - 1)
        fault_nodes_list = combination_fault_list[pick_index]
    else:
        # Will be used when we do concat_Test
        fault_nodes_list = fault_list

    # print("Fault nodes are", list(fault_nodes_list))
    return broadcast(count, commit, number_of_fault_nodes, graph, round_dict, fault_nodes_list, trusted_nodes)


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
                elif current_node == 0 or (trusted_nodes is not None and current_node in trusted_nodes.keys()):
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


##############################
# Main function needed to be called
##############################

# This will generate a graph that satsiifies totality of validity of the graph
def test_graph_main():
    while True:
        print("********************************************************************")
        print(num_edges)
        # Build random Graph
        # graph = build_random_graph()

        # Build random_geo_graph
        graph = build_random_geo_graph()

        # If we find one, break out
        if check_valid(graph):
            break

    # Save to pickles for later usage
    print("Save to pickels")
    pickle.dump(graph, open(f"{TOTAL_NODES}_{THRESHOLD}node_bin_1{int(time())}.p", "wb"))


# Do the broadcast algorithm and also draw out the graph
def broadcast_main(fault_list= None, trusted_nodes= None):
    # Input the graph data saved in pickle to generate the totality validity graph
    graph = pickle.load(open("100_17_100_0.3_200_0.2_300_0.17.p", "rb"))

    print("Graph_Edges:", graph.edges)
    round_dict = broadcast_entry(graph, fault_list, trusted_nodes)
    print("Number of edges", len(list(graph.edges())))
    # Draw the graph
    from Gui import Sim
    Sim(g=graph, d=round_dict)


# Check a single graph validity for debugging purpose when refactor
def check_current_graph_valid_from_file():
    graph = pickle.load(open("50_10node_bin_11551278236.p", "rb"))
    print(check_valid(graph))
    nx.draw(graph)
    plt.show()


def check_current_graph_valid(graph):
    print(check_valid(graph))
    nx.draw(graph)
    plt.show()


# Generate 5 different graph under same parameters
def run_five_times():
    for i in range (5):
        test_graph_main()


##############################
# Concat graph together (not arbitrary bad node)
##############################

# Move your original node id by concat_graph_nodes_number
def move_index(graph, num_current_graph_nodes):
    num_sub_graph_node = len(graph.nodes)
    mapping = dict()
    for i in range(num_sub_graph_node):
        mapping[i] = i + num_current_graph_nodes
    # Relabel the node id with the corresponding id
    nx.relabel_nodes(graph, mapping, copy = False)


# Compose the Two graphs together but are now disconnected component
def compose (src_graph, sub_graph):
    return nx.compose(src_graph, sub_graph)


# Help to prepare the concatination of different graphs together
# fault_nodes_list: the origin id of your bad nodes in individual graph
def concat_build_helper(num_current_graph_nodes, fault_list, concat_fault_node_list, pickle_file_name):
    # Load the graph
    graph = pickle.load(open(pickle_file_name, "rb"))
    # Specify the id for bad nodes in concat graph
    concat_fault_node_list.extend([num_current_graph_nodes + fault_list[0],
                                   num_current_graph_nodes + fault_list[1],
                                   num_current_graph_nodes + fault_list[2]])
    # If not the src graph, need to move its id by num_current_graph_nodes
    if num_current_graph_nodes != 0:
        move_index(graph, num_current_graph_nodes)

    return concat_fault_node_list, graph


# Connect_id is the original sub_graph node
# src_good_list is from the src node
def manual_connect(src_good_list, graph, connect_id):
    for i in range (len(src_good_list)):
        graph.add_edge(src_good_list[i], connect_id)

# Main function for concatinating graphs and run the broadcast
# Note: fault nodes are not arbitrary
# Note: all the previous source node will act like a trusted node


def concat_graph_main():
    num_current_graph_nodes = 0
    concat_fault_list = []
    trusted_nodes = dict()
    # 1
    concat_fault_list, src_graph = concat_build_helper(num_current_graph_nodes, [82, 70, 28],
                                                       concat_fault_list, "100_17_node_bin_11551281716.p")
    num_current_graph_nodes = len(src_graph.nodes)
    # 2
    concat_fault_list, sub_graph_1 = concat_build_helper(num_current_graph_nodes, [5, 58, 78],
                                                         concat_fault_list, "100_0.3_node_geo.p")
    src_graph = compose(src_graph, sub_graph_1)
    num_current_graph_nodes = len(src_graph.nodes)
    manual_connect([4, 31, 48, 94], src_graph, num_current_graph_nodes - 100)
    trusted_nodes[100] = True

    #3
    concat_fault_list, sub_graph_1 = concat_build_helper(num_current_graph_nodes, [122, 164, 189],
                                                         concat_fault_list, "200_0.2_node_geo")
    src_graph = compose(src_graph, sub_graph_1)
    num_current_graph_nodes = len(src_graph.nodes)
    manual_connect([140, 29, 120, 5], src_graph, num_current_graph_nodes - 200)
    trusted_nodes[200] = True

    #4
    concat_fault_list, sub_graph_1 = concat_build_helper(num_current_graph_nodes, [99, 260, 236],
                                                         concat_fault_list, "300_0.17_node_geo")
    src_graph = compose(src_graph, sub_graph_1)
    num_current_graph_nodes = len(src_graph.nodes)
    manual_connect([20, 89, 150, 247], src_graph, num_current_graph_nodes - 300)
    trusted_nodes[400] = True

    # Further build maybe... but follow the previous pattern
    # pickle.dump(src_graph, open("100_17_100_0.3_200_0.2_300_0.17.p", "wb"))
    broadcast_main(concat_fault_list, trusted_nodes)


def main():
    # check_current_graph_valid()
    # test_graph_main()
    concat_graph_main()
    # broadcast_main()


if __name__ == '__main__':
    main()