import os
from random import shuffle
from itertools import combinations
from simulation import load_file_to_graph

rootdir = os.path.join(os.getcwd(), 'subgraphs')


def batch_load_file_to_graph(num_of_G, num_of_g_per_G,
                             g_type=['geo'], num_of_nodes_per_g=[300, 400],
                             graph_param='all', arrangement='random'):
    """
    A generator function that provides an iterator of a (G, [g, ...], {t, ...}) tuple created by load_file_to_graph()
    :param num_of_G: the number of "concated" graphs need to be generated, if = 0, then generates all combinations
    :param num_of_g_per_G: the number of "subgraphs" in a "concated" graph
    :param g_type: the type of the subgraph, e.g. "geo"
    :param num_of_nodes_per_g:
    :param graph_param:
    :param arrangement:
    :return:
    """
    graph_paths = []
    for root, subdirs, files in os.walk(rootdir):
        files = filter(lambda x: x.split('_')[0] in g_type, files)  # filter graph of that type
        files = filter(lambda x: int(x.split('_')[1]) in num_of_nodes_per_g, files)  # filter graph with that many nodes
        if graph_param != 'all':
            files = filter(lambda x: float(x.split('_')[2]) in graph_param, files)  # filter graph with that param
        files = map(lambda x: os.path.join(root, x), files)  # add full path to the file name
        graph_paths.extend(files)

    if len(graph_paths) < num_of_g_per_G:
        raise Exception(f"subgraphs are not enough, found {len(graph_paths)} but needs at least {num_of_g_per_G}")

    if arrangement == 'sorted':
        graph_paths = sorted(graph_paths)
    elif arrangement == 'random':
        shuffle(graph_paths)

    if num_of_G == 0:
        for paths in combinations(graph_paths, num_of_g_per_G):
            yield load_file_to_graph(list(paths))
    else:
        graph_gen_counter = 0
        for paths in combinations(graph_paths, num_of_g_per_G):
            if graph_gen_counter >= num_of_G: return
            yield load_file_to_graph(list(paths))
            graph_gen_counter += 1


def rename():
    path = '/Users/haochen/Desktop/Broadcast_py/subgraphs/geo_100'
    for root, subdirs, files in os.walk(path):
        for file in files:
            if 'n_100' in file:
                os.rename(os.path.join(root, file),
                          os.path.join(root, f'geo_100_{file.split("_")[6]}_{file.split("_")[7]}'))


if __name__ == '__main__':
    # for G, g_list, t_set in batch_load_file_to_graph(5, 3):
    #     print(len(G.nodes))
    rename()
