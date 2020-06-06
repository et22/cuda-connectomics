#include <iostream>
#include <fstream>
#include <bits/stdc++.h> 
using namespace std; 

//Graph class for CPU implementation
// CPU connected components implementation with DFs
//from https://www.geeksforgeeks.org/program-to-count-number-of-connected-components-in-an-undirected-graph/
class Graph { 
    // No. of vertices 
    int V; 
  
    // Pointer to an array containing adjacency lists 
    list<int>* adj; 
  
    // A function used by DFS 
    void DFSUtil(int v, bool visited[]); 
  
public: 
    // Constructor 
    Graph(int V); 
  
    void addEdge(int v, int w); 
    int NumberOfconnectedComponents(); 
}; 
  
int Graph::NumberOfconnectedComponents() 
{ 
  
    // Mark all the vertices as not visited 
    bool* visited = new bool[V]; 
  
    // To store the number of connected components 
    int count = 0; 
    for (int v = 0; v < V; v++) 
        visited[v] = false; 
  
    for (int v = 0; v < V; v++) { 
        if (visited[v] == false) { 
            DFSUtil(v, visited); 
            count += 1; 
        } 
    } 
  
    return count; 
} 
  
void Graph::DFSUtil(int v, bool visited[]) 
{ 
  
    // Mark the current node as visited 
    visited[v] = true; 
  
    // Recur for all the vertices 
    // adjacent to this vertex 
    list<int>::iterator i; 
  
    for (i = adj[v].begin(); i != adj[v].end(); ++i) 
        if (!visited[*i]) 
            DFSUtil(*i, visited); 
} 
  
Graph::Graph(int V) 
{ 
    this->V = V; 
    adj = new list<int>[V]; 
} 
  
// Add an undirected edge 
void Graph::addEdge(int v, int w) 
{ 
    adj[v].push_back(w); 
    adj[w].push_back(v); 
} 

//kernel functions
__global__ void bfs(int* vertices, int* edges, bool* frontier, bool* next_frontier, bool* visited)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if(frontier[tid]){
      frontier[tid] = false;
      visited[tid] = true;                
      for(int i=vertices[tid]; i<vertices[tid+1]; i++){
          int vtx = edges[i];
          if(!visited[vtx]){
              next_frontier[vtx] = true;
          }
      }    
  }
}

__global__ void no_memcpy_bfs(int* vertices, int* edges, bool* frontier, bool* next_frontier, bool* visited)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if(frontier[tid]){
      frontier[tid] = false;
      visited[tid] = true;                
      for(int i=vertices[tid]; i<vertices[tid+1]; i++){
          int vtx = edges[i];
          if(!visited[vtx]){
              next_frontier[vtx] = true;
          }
      }    
  }
}

__global__ void next_to_visit(bool* visited, int* tovisit){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(!visited[tid]){
        tovisit[0] = tid;
    }
}

__global__ void new_frontier(int* tovisit, bool* frontier){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid==tovisit[0]){
        frontier[tovisit[0]] = true;
    }
}

__global__ void check_frontier(bool* frontier, bool* checkfrontier){
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(frontier[tid]){
        checkfrontier[0] = true;
    }
}


//Final GPU solver
void Test_GPU_Solver_BFS(char* arg1, char*arg2)
{ 
    FILE *fdata = fopen(arg2, "r");
    int num_vertex, num_edge;
    char str[100];

    fgets(str,99, fdata);
    sscanf(str, "%d %d", &num_vertex, &num_edge);
    //printf("vetex: %d, edges %d\n", num_vertex, num_edge);
    cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
    cudaEventRecord(start);

    //host vectors
    int* vertices = new int[num_vertex+1];
    int* edges = new int[num_edge];
    bool* frontier = new bool[num_vertex];
    bool* visited = new bool[num_vertex];
    //fill frontier and visited with false
    for(int i=0; i<num_vertex; i++){
        frontier[i]=false;
        visited[i]=false;
    }

    //parse input
    FILE *f = fopen(arg1, "r");
    int row, col;
    int prevrow = -1;
    float value;
    int vertex_count = 0;
    int edge_count = 0;
    while(fgets(str, 99, f)){
        sscanf(str, "%d %d %f", &row, &col, &value);
        if(prevrow!=row){
            vertices[vertex_count] = edge_count;
            vertex_count++;
        }
        edges[edge_count] = col;
        edge_count++;
        prevrow = row;
    }
    vertices[vertex_count] = edge_count;
    //printing vertices and edges arrays
    /*
    printf("\nvertices array: ");
    for(int i=0; i<vertex_count+1; i++){
        printf("%d ", vertices[i]);
    }

    printf("\nedges array: ");
    for(int i=0; i<edge_count; i++){
        printf("%d ", edges[i]);    
    }*/
    //copying vertices, edges, visited from host to device
    int* vertices_on_dev=0;
    int* edges_on_dev=0;
    bool* frontier_b_on_dev=0;
    bool* frontier_a_on_dev=0;
    bool* visited_on_dev=0;

    cudaMalloc((void**)&vertices_on_dev, (num_vertex+1)*sizeof(int));
    cudaMalloc((void**)&edges_on_dev, num_edge*sizeof(int));
    cudaMalloc((void**)&frontier_a_on_dev, num_vertex*sizeof(bool));
    cudaMalloc((void**)&frontier_b_on_dev, num_vertex*sizeof(bool));
    cudaMalloc((void**)&visited_on_dev, num_vertex*sizeof(bool));
    
    cudaMemcpy(vertices_on_dev, vertices,(num_vertex+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(edges_on_dev, edges, num_edge*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(frontier_b_on_dev, frontier, num_vertex*sizeof(bool),cudaMemcpyHostToDevice);
    cudaMemcpy(visited_on_dev, frontier, num_vertex*sizeof(bool),cudaMemcpyHostToDevice);

    //calculating block dim and grid dim based on vertex count
    //large dataset factorization - 3 * 5 * 5 * 11 * 31 * 41
    dim3 blocks(num_vertex, 1, 1);
    dim3 threadsPerBlock(1,1,1); 
    //main bfs algorithm 
    int next_unvisited_idx = 0;
    bool finished = false;
    int number_of_ccs = 0;
    int print_count = 0;
    while(!finished){
        for(int i=std::min(next_unvisited_idx,num_vertex-1); i<num_vertex; i++){
            if(!visited[i]){
                frontier[i] = true;
                next_unvisited_idx = i+1;
                if(print_count>100){
                    double progress = (100.0*i)/((double)num_vertex);
                    printf("Progress: %.2f%%\n", progress);
                    print_count=0;
                }
                print_count++;
                break;
            }
            if(i==num_vertex-1){
                finished=true;
            }
        }
        if(finished) break;
        number_of_ccs++;

        //copy new frontier array to device
        cudaMemcpy(frontier_a_on_dev, frontier, num_vertex*sizeof(bool),cudaMemcpyHostToDevice);

        int count = 0;
        bool is_frontier = true;
        while(is_frontier){
            if(count%2==0){
                //call kernel
                bfs<<<blocks,threadsPerBlock>>>(vertices_on_dev, edges_on_dev, frontier_a_on_dev, frontier_b_on_dev, visited_on_dev);
                cudaMemcpy(frontier, frontier_b_on_dev, num_vertex*sizeof(bool),cudaMemcpyDeviceToHost);
            }
            else{
                //call kernel
                bfs<<<blocks,threadsPerBlock>>>(vertices_on_dev, edges_on_dev, frontier_b_on_dev, frontier_a_on_dev, visited_on_dev);
                cudaMemcpy(frontier, frontier_a_on_dev, num_vertex*sizeof(bool),cudaMemcpyDeviceToHost);
            }

            for(int i=0; i<num_vertex; i++){
                if(frontier[i])
                    break;
                if(i==num_vertex-1)
                    is_frontier = false;
            }

            count++;
        }

        cudaMemcpy(visited,visited_on_dev, num_vertex*sizeof(bool),cudaMemcpyDeviceToHost);
    }
    //printf("\nNumber of CCs: %d", number_of_ccs);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
    //printf("\nGPU runtime: %.4f ms\n",gpu_time);
    printf("%.4f, ", gpu_time);
	cudaEventDestroy(start);
    cudaEventDestroy(end);
}

//Failed optimizations GPU solver
void Test_GPU_Solver_BFS_reduce_memcpy(char* arg1, char*arg2)
{ 
    FILE *fdata = fopen(arg2, "r");
    int num_vertex, num_edge;
    char str[100];

    fgets(str,99, fdata);
    sscanf(str, "%d %d", &num_vertex, &num_edge);
    printf("vetex: %d, edges %d", num_vertex, num_edge);
    cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
    cudaEventRecord(start);

    //host vectors
    int* vertices = new int[num_vertex+1];
    int* edges = new int[num_edge];
    bool* frontier = new bool[num_vertex];
    bool* visited = new bool[num_vertex];
    //fill frontier and visited with false
    for(int i=0; i<num_vertex; i++){
        frontier[i]=false;
        visited[i]=false;
    }
    //parse input
    FILE *f = fopen(arg1, "r");
    int row, col;
    int prevrow = -1;
    float value;
    int vertex_count = 0;
    int edge_count = 0;
    while(fgets(str, 99, f)){
        sscanf(str, "%d %d %f", &row, &col, &value);
        if(prevrow!=row){
            vertices[vertex_count] = edge_count;
            vertex_count++;
        }
        edges[edge_count] = col;
        edge_count++;
        prevrow = row;
    }
    vertices[vertex_count] = edge_count;
    //printing vertices and edges arrays
    /*
    printf("\nvertices array: ");
    for(int i=0; i<vertex_count+1; i++){
        printf("%d ", vertices[i]);
    }

    printf("\nedges array: ");
    for(int i=0; i<edge_count; i++){
        printf("%d ", edges[i]);    
    }*/
    dim3 blocks(num_vertex, 1, 1);
    dim3 threadsPerBlock(1,1,1); 

    //false array
    bool false_array[1];
    false_array[0] = false;

    int zero_array[1];
    zero_array[0] = 0;

    //copying vertices, edges, visited from host to device
    int* vertices_on_dev=0;
    int* edges_on_dev=0;
    bool* frontier_b_on_dev=0;
    bool* frontier_a_on_dev=0;
    bool* visited_on_dev=0;

    bool* checkfrontier=0;
    int* visitnext=0;

    cudaMalloc((void**)&vertices_on_dev, (num_vertex+1)*sizeof(int));
    cudaMalloc((void**)&edges_on_dev, num_edge*sizeof(int));
    cudaMalloc((void**)&frontier_a_on_dev, num_vertex*sizeof(bool));
    cudaMalloc((void**)&frontier_b_on_dev, num_vertex*sizeof(bool));
    cudaMalloc((void**)&visited_on_dev, num_vertex*sizeof(bool));

    cudaMalloc((void**)&checkfrontier, sizeof(bool));
    cudaMalloc((void**)&visitnext, sizeof(bool));
    
    cudaMemcpy(vertices_on_dev, vertices,(num_vertex+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(edges_on_dev, edges, num_edge*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(frontier_b_on_dev, frontier, num_vertex*sizeof(bool),cudaMemcpyHostToDevice);
    cudaMemcpy(visited_on_dev, frontier, num_vertex*sizeof(bool),cudaMemcpyHostToDevice);

    cudaMemcpy(checkfrontier, false_array, sizeof(bool),cudaMemcpyHostToDevice);
    cudaMemcpy(visitnext, zero_array, sizeof(int),cudaMemcpyHostToDevice);
   
    //calculating block dim and grid dim based on vertex count
    //main bfs algorithm 
    int next_unvisited_idx = 0;
    bool finished = false;
    int number_of_ccs = 0;
    int print_count = 0;

    while(!finished){
        for(int i=std::min(next_unvisited_idx,num_vertex-1); i<num_vertex; i++){
            if(!visited[i]){
                frontier[i] = true;
                next_unvisited_idx = i+1;
                if(print_count>100){
                    double progress = (100.0*i)/((double)num_vertex);
                    printf("Progress: %.2f%%\n", progress);
                    print_count=0;
                }
                print_count++;
                break;
            }
            if(i==num_vertex-1){
                finished=true;
            }
        }
        if(finished) break;
        /*
        cudaMemcpy(visited,visited_on_dev, num_vertex*sizeof(bool),cudaMemcpyDeviceToHost);
        printf("\n");
        for(int i=0; i<2000; i++){
                printf("%d ", visited[i]);
        }
        printf("\n");
        next_to_visit<<<blocks,threadsPerBlock>>>(visited_on_dev,visitnext);
        cudaDeviceSynchronize();
        if(print_count>100){
            next_unvisited_idx+=.05;
            printf("approximate progress: %.2f%%\n", next_unvisited_idx);
            print_count = 0; 
        }
        print_count++;
        cudaMemcpy(current_visit_next, visitnext, sizeof(int),cudaMemcpyDeviceToHost);
        printf("current visit next: %d, previous visit next %d\n",current_visit_next[0], previous_visit_next);
        if(current_visit_next[0]==previous_visit_next){
            finished = true;
            break;
        }
        previous_visit_next = current_visit_next[0];

        new_frontier<<<blocks,threadsPerBlock>>>(visitnext, frontier_a_on_dev);
        cudaDeviceSynchronize();
        */
        number_of_ccs++;

        //copy new frontier array to device
        int count = 0;
        bool is_frontier[1];
        is_frontier[0] = true;

        while(is_frontier[0]){
            bool lastcall;
            lastcall= false;
            if(count%2==0){
                //call kernel
                bfs<<<blocks,threadsPerBlock>>>(vertices_on_dev, edges_on_dev, frontier_a_on_dev, frontier_b_on_dev, visited_on_dev);
            }
            else{
                //call kernel
                bfs<<<blocks,threadsPerBlock>>>(vertices_on_dev, edges_on_dev, frontier_b_on_dev, frontier_a_on_dev, visited_on_dev);
                lastcall=true;
            }
            if(lastcall){
                check_frontier<<<blocks,threadsPerBlock>>>(frontier_a_on_dev, checkfrontier);
            }
            else{
                check_frontier<<<blocks,threadsPerBlock>>>(frontier_b_on_dev, checkfrontier);
            } 
            cudaDeviceSynchronize();
            cudaMemcpy(is_frontier, checkfrontier, sizeof(bool),cudaMemcpyDeviceToHost);
            cudaMemcpy(checkfrontier, false_array, sizeof(bool),cudaMemcpyHostToDevice);
            count++;
        }
        cudaMemcpy(visited,visited_on_dev, num_vertex*sizeof(bool),cudaMemcpyDeviceToHost);
    }
    printf("\nNumber of CCs: %d", number_of_ccs);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
    printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
    cudaEventDestroy(end);
}

//CPU solver
void Test_CPU_Solver_BFS(char* arg1, char*arg2){
    cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
    cudaEventRecord(start);

    FILE *fdata = fopen(arg2, "r");
    int num_vertex, num_edge;
    char str[100];

    fgets(str,99, fdata);
    sscanf(str, "%d %d", &num_vertex, &num_edge);
    //printf("vetex: %d, edges %d\n", num_vertex, num_edge);

    Graph g(num_vertex);

    //parse input
    FILE *f = fopen(arg1, "r");
    int row, col;
    float value;
    while(fgets(str, 99, f)){
        sscanf(str, "%d %d %f", &row, &col, &value);
        g.addEdge(row, col);
    }

    int number_of_ccs = g.NumberOfconnectedComponents();
    //printf("\nNumber of CCs: %d", number_of_ccs);
    cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
    //printf("\nCPU runtime: %.4f ms\n",gpu_time);
    printf("%.4f, ", gpu_time);
	cudaEventDestroy(start);
    cudaEventDestroy(end);
}

int main(int argc, char* argv[])
{
    printf("Testing GPU solver:\n");
    for(int i=0; i<20; i++){
        Test_GPU_Solver_BFS(argv[1], argv[2]);
    }
    printf("\nTesting CPU solver:\n");
    for(int i=0; i<20; i++){
        Test_CPU_Solver_BFS(argv[1], argv[2]);
    }
	return 0;
}