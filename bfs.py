# Ethan Trepka
# BFS functions
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

def bfs_on_gpu_v1(graph_dictionary, number_of_edges):
    # create vertices to int mappings
    vertex_count = 0
    key_count = {}
    count_key = {}
    for key in graph_dictionary:
        if key not in key_count:
            key_count[key] = vertex_count
            count_key[vertex_count] = key
            vertex_count += 1
        """
        for edge in graph_dictionary[key]:
            if edge not in key_count:
                key_count[edge] = vertex_count
                count_key[vertex_count] = edge
                vertex_count += 1
        """

    vertices_array = np.zeros(vertex_count + 1, dtype=int)
    edges_array = np.zeros(number_of_edges, dtype=int)
    frontier_array = np.zeros(vertex_count, dtype=bool)
    next_frontier_array = np.zeros(vertex_count, dtype=bool)
    visited_array = np.zeros(vertex_count, dtype=bool)
    x = np.arange(vertex_count, dtype=int)
    cost_array = np.full_like(x, 100000000)

    debug_array = np.zeros(vertex_count + 1, dtype=int)


    vertices_count = 0
    edges_count = 0
    for key in graph_dictionary:
        vertices_array[vertices_count] = edges_count
        for edge in graph_dictionary[key]:
            edges_array[edges_count] = key_count[edge]
            edges_count += 1
        vertices_count += 1
    vertices_array[vertices_count] = edges_count

    vertices_array_gpu = cuda.mem_alloc(vertices_array.size * vertices_array.dtype.itemsize)
    edges_array_gpu = cuda.mem_alloc(edges_array.size*edges_array.dtype.itemsize)
    frontier_array_gpu = cuda.mem_alloc(frontier_array.size*frontier_array.dtype.itemsize)
    next_frontier_array_gpu = cuda.mem_alloc(next_frontier_array.size*next_frontier_array.dtype.itemsize)
    cost_array_gpu = cuda.mem_alloc(cost_array.size*cost_array.dtype.itemsize)
    visited_array_gpu = cuda.mem_alloc(visited_array.size*visited_array.dtype.itemsize)
    debug_array_gpu = cuda.mem_alloc(debug_array.size*debug_array.dtype.itemsize)


    # all the pycuda code below
    # set source before memcpy
    frontier_array[0] = 1
    cost_array[0] = 0
    cuda.memcpy_htod(vertices_array_gpu, vertices_array)
    cuda.memcpy_htod(edges_array_gpu, edges_array)
    cuda.memcpy_htod(frontier_array_gpu, frontier_array)
    cuda.memcpy_htod(next_frontier_array_gpu, next_frontier_array)
    cuda.memcpy_htod(cost_array_gpu, cost_array)
    cuda.memcpy_htod(visited_array_gpu, visited_array)
    cuda.memcpy_htod(debug_array_gpu, debug_array)


    mod = SourceModule("""
      __global__ void bfs(int* vertices, int* edges, bool* frontier, bool* next_frontier, bool* visited, int* cost, int* debug)
      {
        int tid = threadIdx.x;
        debug[tid]=1;
        if(frontier[tid]){
            frontier[tid] = false;
            visited[tid] = true;
            //int i = vertices[tid];
            //int j = vertices[tid+1];
            //debug[tid] = i;
            //debug[tid+1]=j;
                      
            for(int i=vertices[tid]; i<vertices[tid+1]; i++){
                int vtx = edges[i];
                if(!visited[vtx]){
                    cost[vtx] = cost[tid]+1;
                    next_frontier[vtx] = true;
                }
            }
            
        }
      }
      """)
    components_array = np.zeros(vertex_count, dtype=int)
    # debugging
    func = mod.get_function("bfs")

    while sum(visited_array) < len(visited_array):
        iterations = 0
        while sum(frontier_array) > 0:
            print(iterations)
            if iterations % 2 == 0:
                func(vertices_array_gpu, edges_array_gpu, frontier_array_gpu, next_frontier_array_gpu, visited_array_gpu,
                     cost_array_gpu, debug_array_gpu,
                     block=(vertices_count, 1, 1))
                cuda.Context.synchronize()
                cuda.memcpy_dtoh(frontier_array, next_frontier_array_gpu)
                cuda.memcpy_dtoh(next_frontier_array, frontier_array_gpu)
            else:
                func(vertices_array_gpu, edges_array_gpu, next_frontier_array_gpu, frontier_array_gpu, visited_array_gpu,
                     cost_array_gpu, debug_array_gpu,
                     block=(vertices_count, 1, 1))
                cuda.Context.synchronize()
                cuda.memcpy_dtoh(frontier_array, frontier_array_gpu)
                cuda.memcpy_dtoh(next_frontier_array, next_frontier_array_gpu)
            iterations += 1
            cuda.memcpy_dtoh(vertices_array,vertices_array_gpu)
            cuda.memcpy_dtoh(debug_array, debug_array_gpu)
            cuda.Context.synchronize()
            print("debug array")
            print(debug_array)
        cuda.memcpy_dtoh(visited_array, visited_array_gpu)
        cuda.memcpy_dtoh(cost_array, cost_array_gpu)
        print("visibed before")
        print(visited_array)
        for j in range(len(visited_array)):
            components_array[j] += visited_array[j]
        for i in range(len(visited_array)):
            if visited_array[i] == 0:
                frontier_array[i] = 1
                cost_array[i] = 0
                cuda.memcpy_htod(frontier_array_gpu, frontier_array)
                cuda.memcpy_htod(cost_array_gpu, cost_array)
                break
    return components_array_to_dictionary(components_array)


# implementing connected components bfs with cusp
def bfs_on_gpu_cusp(current_dictionary):
    temp = 1


# helper function to convert component array to dictionary
def components_array_to_dictionary(components):
    dictionary = {}
    for i in range(len(components)):
        if components[i] in dictionary:
            dictionary[components[i]].add(i)
        else:
            dictionary[components[i]] = {i}
    return dictionary
