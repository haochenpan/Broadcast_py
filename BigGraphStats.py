import matplotlib.pyplot as plt
import numpy as np
import pickle


# Individual graph
# x_axis: # trused nodes
# y_axis: $ rounds
def line_graph(result_dict, graph_num):
    x_axis = list(range(0, 300, 10))
    # index represents the id of individual algorithm
    # each value represents the list of number of rounds corresponding to number of trusted
    total_list = []

    small_dict = result_dict[0]

    num_alg = len(small_dict.keys())

    for i in range(num_alg):
        total_list.append([])

    for num_trusted, small_dict in result_dict.items():
        for alg_idx, rounds in small_dict.items():
            total_list[alg_idx - 1].append(rounds[graph_num])

    for alg_label in range(len(total_list)):
        rounds_list = total_list[alg_label]
        if alg_label == 0:
            plt.plot(x_axis, rounds_list, 'r', label='CLOSENESS_CENTRALITY')

        elif alg_label == 1:
            plt.plot(x_axis, rounds_list, 'b', label='BETWEENESS_CENTRALITY')

        if alg_label == 2:
            plt.plot(x_axis, rounds_list, 'g', label='DEGREE_CENTRALITY')

        if alg_label == 3:
            plt.plot(x_axis, rounds_list, 'y', label='CLOSENESS_CENTRALITY_REMOVE')

        elif alg_label == 4:
            plt.plot(x_axis, rounds_list, 'c', label='BETWEENESS_CENTRALITY_REMOVE')

        elif alg_label == 5:
            plt.plot(x_axis, rounds_list, 'm', label='DEGREE_CENTRALITY_REMOVE')

    plt.legend(loc='best')
    plt.xlabel("Number of trusted")
    plt.ylabel("Number of rounds")
    plt.title("Graph_id:7")
    plt.xticks(list(range(0,300,10)))
    plt.show()
    # plt.savefig(f"Medium, graph_id:{graph_num}")
    plt.clf()


def line_graph_head(result_dict, graph_num, top_k_trusted):
    x_axis = list(range(0, top_k_trusted + 1, 10))
    # index represents the id of individual algorithm
    # each value represents the list of number of rounds corresponding to number of trusted
    total_list = []

    small_dict = result_dict[0]

    num_alg = len(small_dict.keys())

    for i in range(num_alg):
        total_list.append([])

    for num_trusted, small_dict in result_dict.items():
        if num_trusted <= top_k_trusted:
            for alg_idx, rounds in small_dict.items():
                total_list[alg_idx - 1].append(rounds[graph_num])

    for alg_label in range(len(total_list)):
        rounds_list = total_list[alg_label]
        if alg_label == 0:
            plt.plot(x_axis, rounds_list, 'r', label='CLOSENESS_CENTRALITY')

        elif alg_label == 1:
            plt.plot(x_axis, rounds_list, 'b', label='BETWEENESS_CENTRALITY')

        if alg_label == 2:
            plt.plot(x_axis, rounds_list, 'g', label='DEGREE_CENTRALITY')

        if alg_label == 3:
            plt.plot(x_axis, rounds_list, 'y', label='CLOSENESS_CENTRALITY_REMOVE')

        elif alg_label == 4:
            plt.plot(x_axis, rounds_list, 'c', label='BETWEENESS_CENTRALITY_REMOVE')

        elif alg_label == 5:
            plt.plot(x_axis, rounds_list, 'm', label='DEGREE_CENTRALITY_REMOVE')

    plt.legend(loc='best')
    plt.xlabel("Number of trusted")
    plt.ylabel("Number of rounds")
    plt.title(f"Graph_id:7")
    plt.xticks(list(range(0, top_k_trusted + 1, 10)))
    plt.yticks(list(range(10, 36,5)))
    plt.show()
    # plt.savefig(f"Medium, graph_id:{graph_num}")
    plt.clf()



def plotOne(fileName):
    result_dict = pickle.load(open(fileName, "rb"))
    line_graph(result_dict, 0)
    line_graph_head(result_dict, 0, 50)

def main():
    fileName = "/Users/yingjianwu/Desktop/broadcast/Broadcast_py/Big_Graph_7.p"
    plotOne(fileName)

    # result_dict = dict()
    # result_list = []
    # for i in range(12):
    #     local_dict = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/result_dict_0123.pickle", "rb"))
    #     result_list.append(local_dict)
    #
    # result_dict = result_dict[0]
    #
    # for i in range(1, len(result_dict)):
    #     add_dict = result_list[i]
    #     for num_trusted, small_dict in result_dict.items():
    #
    #         for alg_num in small_dict.keys():
    #
    #             small_dict[alg_num].extend(add_dict[num_trusted][alg_num])
    #
    # graph_id = 0
    # line_graph(result_dict, graph_id)



# key: number of trusted
# value: small dict:
    # key: algorithm name
    # number of rounds

if __name__ == '__main__':
    main()


