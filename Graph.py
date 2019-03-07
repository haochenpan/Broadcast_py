import matplotlib.pyplot as plt
import numpy as np
import pickle


def preprocess(para_list, dic):
    rounds_data_list = []
    for para in para_list:
        rounds_data_list.append(dic[para])
    return rounds_data_list


def std_avg_graph():
    para_u = [0, 6, 12, 24, 48, 96, 188]
    para_r = [0, 16, 32, 48, 64, 96, 128, 256, 512, 768]

    u_data_dict = pickle.load(open("uni_data_1000_", "rb"))
    r_data_dict = pickle.load(open("ratio_data_1000_", "rb"))

    data_to_plot_u = preprocess(para_u, u_data_dict)

    data_to_plot_r = preprocess(para_r, r_data_dict)

    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot_r, showmeans=True, labels=para_r)
    plt.xlabel("Number_Of_Trusted_Nodes")
    plt.ylabel("Number of Rounds")
    plt.title("Ratio_Distributed_Good_Nodes_In_Random_Case")
    plt.show()


def alg_compare_graph(result_dict):
    for num_trusted, small_dict in result_dict.items():
        for alg_label, rounds_list in small_dict.items():
            if alg_label == 0:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'r', label='DEGREE_CENTRALITY')
            elif alg_label == 1:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'b', label='EIGEN_CENTRALITY')
            elif alg_label == 2:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'g', label='CLOSENESS_CENTRALITY')
            elif alg_label == 3:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'y', label='BETWEENNESS_CENTRALITY')
            elif alg_label == 4:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'c', label='UNIFORM_TOTAL')
            elif alg_label == 5:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'm', label='UNIFORM_SUB')
            elif alg_label == 6:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'k', label='WEIGHTED_EDGEs')
            elif alg_label == 7:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'tab:purple', label='remove_neis')
            elif alg_label == 8:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'tab:purple', label='remove_c_clo')
            elif alg_label == 9:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'tab:purple', label='remove_c_bet')
            elif alg_label == 10:
                plt.plot(list(range(len(rounds_list))), rounds_list, 'tab:purple', label='remove_c_Edg')

        plt.xticks(np.arange(0, len(rounds_list), 1.0))
        plt.legend(loc='best')
        plt.title(f"number of trusted node : {num_trusted}")
        plt.savefig(f"testing_int{num_trusted}")
        plt.clf()

#
# def bar_graph():


# Individual graph
# x_axis: # trused nodes
# y_axis: $ rounds
def line_graph(result_dict, graph_num):
    x_axis = []
    for num_trusted in result_dict.keys():
        x_axis.append(num_trusted)

    # index represents the id of individual algorithm
    # each value represents the list of number of rounds corresponding to number of trusted
    total_list = []

    small_dict = result_dict[0]

    num_alg = len(small_dict.keys())

    for i in range(num_alg):
        total_list.append([])

    for num_trusted, small_dict in result_dict.items():
        for alg_idx, rounds in small_dict.items():
            total_list[alg_idx].append(rounds[graph_num])

    for alg_label in range(len(total_list)):
        rounds_list = total_list[alg_label]
        if alg_label == 0:
            plt.plot(x_axis, rounds_list, 'r', label='DEGREE_CENTRALITY')

        elif alg_label == 1:
            plt.plot(x_axis, rounds_list, 'b', label='EIGEN_CENTRALITY')

        if alg_label == 2:
            plt.plot(x_axis, rounds_list, 'g', label='CLOSENESS_CENTRALITY')

        if alg_label == 3:
            plt.plot(x_axis, rounds_list, 'y', label='BETWEENNESS_CENTRALITY')

        elif alg_label == 4:
            plt.plot(x_axis, rounds_list, 'c', label='UNIFORM_TOTAL')

        elif alg_label == 5:
            plt.plot(x_axis, rounds_list, 'm', label='UNIFORM_SUB')

        elif alg_label == 6:
            plt.plot(x_axis, rounds_list, 'k', label='WEIGHTED_EDGEs')

        elif alg_label == 7:
            plt.plot(x_axis, rounds_list, 'tab:pink', label='remove_c_clo')

        elif alg_label == 8:
            plt.plot(x_axis, rounds_list, 'tab:orange', label='remove_c_bet')

        elif alg_label == 9:
            plt.plot(x_axis, rounds_list, 'tab:gray', label='remove_c_Degree')

    plt.legend(loc='best')
    plt.xlabel("Number of trusted")
    plt.ylabel("Number of rounds")
    plt.title(f"Medium, graph_id:{graph_num}")
    # plt.show()
    plt.savefig(f"Medium, graph_id:{graph_num}")
    plt.clf()


# Individual graph
# x_axis: # trused nodes
# y_axis: $ rounds
def line_graph_head(result_dict, graph_num, top_k_trusted):
    x_axis = []
    for num_trusted in result_dict.keys():
        if num_trusted <= top_k_trusted:
            x_axis.append(num_trusted)

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
                total_list[alg_idx].append(rounds[graph_num])

    for alg_label in range(len(total_list)):
        rounds_list = total_list[alg_label]
        if alg_label == 0:
            plt.plot(x_axis, rounds_list, 'r', label='DEGREE_CENTRALITY')

        elif alg_label == 1:
            plt.plot(x_axis, rounds_list, 'b', label='EIGEN_CENTRALITY')

        if alg_label == 2:
             plt.plot(x_axis, rounds_list, 'g', label='CLOSENESS_CENTRALITY')

        if alg_label == 3:
            plt.plot(x_axis, rounds_list, 'y', label='BETWEENNESS_CENTRALITY')

        elif alg_label == 4:
            plt.plot(x_axis, rounds_list, 'c', label='UNIFORM_TOTAL')

        elif alg_label == 5:
            plt.plot(x_axis, rounds_list, 'm', label='UNIFORM_SUB')

        elif alg_label == 6:
            plt.plot(x_axis, rounds_list, 'k', label='WEIGHTED_EDGEs')

        elif alg_label == 7:
            plt.plot(x_axis, rounds_list, 'tab:pink', label='remove_c_clo')

        elif alg_label == 8:
            plt.plot(x_axis, rounds_list, 'tab:orange', label='remove_c_bet')

        elif alg_label == 9:
            plt.plot(x_axis, rounds_list, 'tab:gray', label='remove_c_Degree')


    plt.legend(loc='best')
    plt.xticks(np.arange(0, 10, 1.0))
    plt.xlabel("Number of trusted")
    plt.ylabel("Number of rounds")
    plt.title(f"Medium, graph_id:{graph_num}_head_rounds")
    plt.savefig(f"Medium, graph_id:{graph_num}_head_rounds")
    plt.show()
    plt.clf()


def bar_graph(graph_id, result_dict):
    # Title: Number of trusted in graph_id {}

    # x_axis = [algorithm]
    x_axis = ['DEGREE', 'EIGEN', 'CLOSENESS', 'BETWEENNESS', 'U_TOTAL', 'U_sub', 'W_EDGEs', 'REMOVE_NEIS']
    # y_axis = lantency
    bottom = 10
    top = 20
    for num_trusted, small_dic in result_dict.items():
        y_local_list = []
        for alg_name, rounds_list in small_dic.items():
            y_local_list.append(rounds_list[graph_id])
        plt.xlabel("Algorithm_Name")
        plt.ylabel("Latency (rounds)")
        index = np.arange(len(x_axis))
        plt.xticks(index, x_axis, fontsize = 5)
        plt.yticks(np.arange(bottom , top, step = 1))
        plt.ylim(bottom,top)
        plt.bar(x_axis, y_local_list)
        plt.title(f"Medium: Number of trusted {num_trusted} in graph_id {graph_id}")
        # plt.show()
        plt.savefig(f"{num_trusted}_Bar")
        plt.clf()


# key: number of trusted
# value: small dict:
    # key: algorithm name
    # number of rounds
if __name__ == '__main__':
    result_dict = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/result_dict_0123.pickle", "rb"))
    result_dict_456 = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/result_dict_456.pickle", "rb"))
    # result_dict_remove = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/remove_greedy_geo_1.p", "rb"))
    result_dict_common_close = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/remove_greedy_geo_2.p", "rb"))
    result_dict_common_betweeness = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/remove_greedy_geo_3.p", "rb"))
    result_dict_common_edges = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/remove_greedy_geo_4.p", "rb"))

    for k, v in result_dict_456.items():
        small_dict_in_result_dict = result_dict[k]
        for a, b in v.items():
            small_dict_in_result_dict[a] = b

    # for k, v in result_dict_remove.items():
    #     small_dict_in_result_dict = result_dict[k]
    #     for a, b in v.items():
    #         small_dict_in_result_dict[a] = b

    for k, v in result_dict_common_close.items():
        small_dict_in_result_dict = result_dict[k]
        for a, b in v.items():
            small_dict_in_result_dict[a] = b

    for k, v in result_dict_common_betweeness.items():
        small_dict_in_result_dict = result_dict[k]
        for a, b in v.items():
            small_dict_in_result_dict[a] = b

    for k, v in result_dict_common_edges.items():
        small_dict_in_result_dict = result_dict[k]
        for a, b in v.items():
            small_dict_in_result_dict[a] = b

    graph_id = 18
    line_graph(result_dict, graph_id)
    line_graph_head(result_dict, graph_id, 10)
    # bar_graph(graph_id, result_dict)

