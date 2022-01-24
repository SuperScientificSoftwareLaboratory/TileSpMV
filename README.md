# TileSPMV

 

**TileSpMV** is an open source code of the paper:

#-------------------

Y. Niu, Z. Lu, M. Dong, Z. Jin, W. Liu and G. Tan, "TileSpMV: A Tiled Algorithm for Sparse Matrix-Vector Multiplication on GPUs," 2021 IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2021, pp. 68-78, doi: 10.1109/IPDPS49936.2021.00016.


## Introduction

Sparse matrix-vector multiplication(SpMV) executes Ax=y, where A is a sparse matrix, x and y are dense vectors. TileSpMV implemented seven warp-level SpMV methods to calculate sparse tiles stored in different formats, and a format selection method is designed to select the best format and algorithm for each sparse tile to improve performance from the perspective of the local sparse structure of the matrix. In addition, nonzeros in very sparse tiles are extracted into a separate matrix for better performance. 
TileSpMV provides a version of CUDA on a high parallelism currently.


<!-- ## Structure
```
beidoublas/README     instructions on installation
beidoublas/src        C source code, to be compiled into libbeidoublas.so
beidoublas/test       testing code
beidoublas/Makefile   top-level Makefile that does installation and testing
``` -->

## Installation

<!-- To use this code, you need to modify the Makefile with correct g++ installation path and use make for automatic installation. -->
NVIDIA GPU with compute capability at least 3.5 (NVIDIA Tesla K40 as tested) * NVIDIA nvcc CUDA compiler and cuSPARSE library, both of which are included with CUDA Toolkit (CUDA v11.1 as tested) 
The GPU test programs have been tested on Ubuntu 18.04/20.04, and are expected to run correctly under other Linux distributions.

## Execution of TileSpMV
Our test programs currently support input files encoded using the matrix market format. All matrix market datasets used in this evaluation are publicly available from the SuiteSparse Matrix Collection. 

1. Set CUDA path in the Makefile

2. The command 'make' generates an executable file 'test' for double precision.
> **make**

3. Run SpMV code on matrix data with auto-tuning in double precision. The GPU compilation takes an optional d=<gpu-device, e.g., 0> parameter that specifies the GPU device to run if multiple GPU devices are available at the same time. 
> **./test -d 0 test.mtx**



## Release version
Oct 19,2021 Version Alpha

 



