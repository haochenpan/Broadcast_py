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
            elif alg_label == 'remove_greedy_globalView':
                plt.plot(list(range(len(rounds_list))), rounds_list,  'tab:purple', label='remove_neis')

        plt.xticks(np.arange(0, len(rounds_list), 1.0))
        plt.legend(loc='best')
        plt.title(f"number of trusted node : {num_trusted}")
        plt.savefig(f"testing_int{num_trusted}")
        plt.clf()


# key: number of trusted
# value: small dict:
    # key: algorithm name
    # number of rounds
if __name__ == '__main__':
    result_dict = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/result_dict_0123.pickle", "rb"))
    result_dict_456 = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/result_dict_456.pickle", "rb"))
    result_dict_remove = pickle.load(open("/Users/yingjianwu/Desktop/broadcast/Broadcast_py/remove_greedy_geo_1.p", "rb"))

    for k, v in result_dict_456.items():
        small_dict_in_result_dict = result_dict[k]
        for a, b in v.items():
            small_dict_in_result_dict[a] = b

    for k, v in result_dict_remove.items():
        small_dict_in_result_dict = result_dict[k]
        for a, b in v.items():
            small_dict_in_result_dict[a] = b

    alg_compare_graph(result_dict)