from os import listdir
from os.path import isfile, join
from pickle import load
from broadcast_refactor import check_valid, validGraph
from collections import defaultdict

# print('*'*10, "Batch Validating", '*'*10)
# my_path = '/Users/haochen/Desktop/Broadcast_py/bin'
# files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
# for file in files:
#     if '.pi' in file:
#         file_path = my_path + '/' + file
#         graph = load(open(file_path, 'rb'))
#         print(len(graph.edges))
#     # print("current graph", file)
#     # print(check_valid(graph))

print('*'*10, "Single graph Validating", '*'*10)
file_path = '/Users/haochen/Desktop/Broadcast_py/bin/n_60_f_3_geo_th_0.5_1551618764.pi'
graph = load(open(file_path, 'rb'))
print(check_valid(graph))

# print('*'*10, "Single broadcast Validating", '*'*10)
# count = defaultdict(lambda: dict())
# commit = [False] * (len(graph.nodes))
# v = validGraph(count, commit, 3, graph, {11, 9, 3})
# print(v)