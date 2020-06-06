# cuda-connectomics
CUDA, PyCUDA, CUSP, C++ implementations of connected components algorithms to run on MICRONS data. 

## compilation for primary CUDA version
To compile: nvcc -std=c++11 basic_kernel_bfs.cu

To run: `./a.out [file_path_out] [fil_path_data]`

Examples:

test 1: 

`./a.out testfiles/test1 testfiles/test1_data`

microns_small:

`./a.out microns_cleaned_data/small/microns_out.txt microns_cleaned_data/small/microns_out_data.txt`

## compilation for CUSP
`nvcc -std=c++11  cusp_bfs.cu  -I /dartfs-hpc/rc/home/9/USERNAME/`

requires installation of CUSP, google CUSP github for install instructions, must specify cusp library path if not included in CUDA library path as shown above

note: CUSP library is not production tested

## compilation for PyCUDA
`python main.py`

note: current python version is not completely finished
