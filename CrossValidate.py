from os import listdir
from os.path import isfile, join
from pickle import load
from broadcast_refactor import check_valid, validGraph
from collections import defaultdict

my_path = '/home/eaglevisionapp/broadcast/n100/bin'
files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
for file in files:
    file_path = my_path + '/' + file
    graph = load(open(file_path, 'rb'))
    print("current graph", file)
    print(check_valid(graph))

# file_path = '/Users/haochen/Desktop/Broadcast_py/bin_100/n_100_f_3_geo_th_0.3_1551498220.pi'
# graph = load(open(file_path, 'rb'))
# print(check_valid(graph))


# count = defaultdict(lambda: dict())
# commit = [False] * (len(graph.nodes))
# v = validGraph(count, commit, 3, graph, {11, 9, 3})
# print(v)