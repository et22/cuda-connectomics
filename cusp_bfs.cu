#include <iostream>
#include <fstream>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/hyb_matrix.h>
#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/blas/blas.h>
#include <cusp/linear_operator.h>
#include <cusp/gallery/poisson.h>
#include <cusp/convert.h>
#include <cusp/krylov/gmres.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/gallery/grid.h>
//include connected components header file
#include <cusp/graph/connected_components.h>
#include <list> 

void Test_GPU_Solver_CUSP(char* arg1, char*arg2)
{ 
    FILE *fdata = fopen(arg2, "r");
    int num_vertex, num_edge;
    char str[100];

    fgets(str,99, fdata);
    sscanf(str, "%d %d", &num_vertex, &num_edge);

    cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
    cudaEventRecord(start);
    cusp::coo_matrix<int,float,cusp::host_memory> coo_mat(num_vertex,num_vertex,num_edge);

    FILE *f = fopen(arg1, "r");
    int row, col;
    float value;
    int count = 0;
    while(fgets(str, 99, f)){
        sscanf(str, "%d %d %f", &row, &col, &value);
        coo_mat.row_indices[count] = row;
        coo_mat.column_indices[count] = col;
        coo_mat.values[count] = value;
        count++;
    }
    
    cusp::csr_matrix<int,float,cusp::host_memory> csr_mat(coo_mat);
    cusp::csr_matrix<int,float,cusp::device_memory> csr_mat_device(csr_mat);
    cusp::array1d<int,cusp::device_memory> components(csr_mat_device.num_rows);
    size_t numparts = cusp::graph::connected_components(csr_mat_device, components);
    std::cout << "Found " << numparts << " components in the graph." <<std::endl;   

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
    //printf("\nGPU runtime: %.4f ms\n",gpu_time);
    printf("\n%.4f",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

void Test_GPU_Solver_CUSP_BFS(char* arg1, char*arg2)
{ 
    FILE *fdata = fopen(arg2, "r");
    int num_vertex, num_edge;
    char str[100];

    fgets(str,99, fdata);
    sscanf(str, "%d %d", &num_vertex, &num_edge);

    cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
    cudaEventRecord(start);
    cusp::coo_matrix<int,float,cusp::host_memory> coo_mat(num_vertex,num_vertex,num_edge);

    FILE *f = fopen(arg1, "r");
    int row, col;
    float value;
    int count = 0;
    while(fgets(str, 99, f)){
        sscanf(str, "%d %d %f", &row, &col, &value);
        coo_mat.row_indices[count] = row;
        coo_mat.column_indices[count] = col;
        coo_mat.values[count] = value;
        count++;
    }
    
    cusp::csr_matrix<int,float,cusp::host_memory> csr_mat(coo_mat);
    cusp::csr_matrix<int,float,cusp::device_memory> csr_mat_device(csr_mat);
    cusp::array1d<int,cusp::device_memory> labels(csr_mat_device.num_rows);
    // Execute a BFS traversal on the device
    cusp::graph::breadth_first_search(csr_mat_device, 0, labels);
    // Print the level set constructed from the source vertex
    cusp::print(labels);    


	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
    //printf("\nGPU runtime: %.4f ms\n",gpu_time);
    printf("\n%.4f",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

int main(int argc, char* argv[])
{
    //for(int i=0; i<20; i++){
        Test_GPU_Solver_CUSP(argv[1], argv[2]);
    //}
	return 0;
}