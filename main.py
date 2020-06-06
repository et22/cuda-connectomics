# Ethan Trepka
# Final project main function

import time
from bfs import *
from microns_data_preprocessing import *
from test_graphs import *


def print_dictionary(dictionary):
    for key, value in dictionary.items():
        print(key, ': ', value)

"""
# demonstrating correctness
# test case 1
print("Test case 1:")
print("Adjacency list:")
graph_dictionary, numEdges = generate_test_0()
print_dictionary(graph_dictionary)
cc_dictionary = bfs_on_gpu_v1(graph_dictionary, numEdges)
print("Connected components:")
print_dictionary(cc_dictionary)
# test case 2
print("\n\nTest case 2:")
print("Adjacency list:")
graph_dictionary, numEdges = generate_test_2()
print_dictionary(graph_dictionary)
cc_dictionary = bfs_on_gpu_v1(graph_dictionary, numEdges)
print("Connected components:")
print_dictionary(cc_dictionary)

# test case 3
print("\n\nTest case 3:")
print("Adjacency list:")
graph_dictionary, numEdges = generate_test_3()
print_dictionary(graph_dictionary)
cc_dictionary = bfs_on_gpu_v1(graph_dictionary, numEdges)
print("Connected components:")
print_dictionary(cc_dictionary)
"""

# processing simplest microns dataset and loading into dictionary
print("\n\nMicrons Dataset:")
print("Adjacency list:")

file_num = 0
graph_dictionary, numEdges = process_microns_undirected(file_num)

# calling and timing gpu bfs
tic = time.clock()
cc_dictionary = bfs_on_gpu_v1(graph_dictionary, numEdges)
toc = time.clock()
run_time = (toc-tic)/1.0e3
print('Microns small data, GPU bfs time: ' + str(run_time) + 'milliseconds')
print("Connected components:")
print_dictionary(cc_dictionary)
