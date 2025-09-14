#include <device_launch_parameters.h>
#include <cuda_runtime.h> 
#include <cuComplex.h>
using namespace std;


__global__ void P2Mkernel(cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, int, int, int);
__global__ void L2Pkernel(cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, int, int, int);
__global__ void L2Pkernelh(cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, int, int, int);
__global__ void M2Mkernel_output(cuDoubleComplex*, cuDoubleComplex*, int, int, int, double*, int*, int, cuDoubleComplex*);
__global__ void M2Mkernel(cuDoubleComplex*, cuDoubleComplex*, int, double*, int*, int, cuDoubleComplex*);
__global__ void M2Lkernel_output(cuDoubleComplex*, cuDoubleComplex*, int, int, int*, int*, cuDoubleComplex*);
__global__ void P2Pkernel(cuDoubleComplex*, cuDoubleComplex*, int*, cuDoubleComplex*, int chunkSize);
__global__ void L2Lkernel_output(cuDoubleComplex*, cuDoubleComplex*, int, int, int, double*, int*, int, cuDoubleComplex*);
__global__ void L2Lkernelh_output(cuDoubleComplex*, cuDoubleComplex*, int, int, int, double*, int*, int, cuDoubleComplex*);
__global__ void L2Lkernel(cuDoubleComplex*, cuDoubleComplex*, int, double*, int*, int, cuDoubleComplex*);
__global__ void L2Lkernelh(cuDoubleComplex*, cuDoubleComplex*, int, double*, int*, int, cuDoubleComplex*);
__global__ void locate(cuDoubleComplex*, int*, cuDoubleComplex*, int);
__global__ void relocate(cuDoubleComplex*, int*, cuDoubleComplex*, int);
__global__ void norm2partc(double*, cuDoubleComplex*);
__global__ void norm2partd(double*);
__global__ void innerpartc(cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*);
__global__ void innerpartd(cuDoubleComplex*);
__global__ void prep(cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*);
__global__ void prep(cuDoubleComplex*, cuDoubleComplex*);
__global__ void post(cuDoubleComplex*, cuDoubleComplex*);
__global__ void post(cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*);
__global__ void saxp(cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex);

void P2Mkernel(cuDoubleComplex* C, cuDoubleComplex* A, cuDoubleComplex* B, int M, int N, int L, dim3 grid, dim3 block, int mem, cudaStream_t str);
void L2Pkernel(cuDoubleComplex* C, cuDoubleComplex* A, cuDoubleComplex* B, int M, int N, int L, dim3 grid, dim3 block, int mem);
void L2Pkernelh(cuDoubleComplex* C, cuDoubleComplex* A, cuDoubleComplex* B, int M, int N, int L, dim3 grid, dim3 block, int mem);
void M2Mkernel(cuDoubleComplex* agg, cuDoubleComplex* aggl, int sampl, double* interp, int* intind, int ninter, cuDoubleComplex* shift, dim3 grid, dim3 block, int mem, cudaStream_t str);
void M2Mkernel_output(cuDoubleComplex* agg, cuDoubleComplex* aggl, int samp, int sampl, int numclus, double* interp, int* intind, int ninter, cuDoubleComplex* shift, dim3 grid, dim3 block, int mem, cudaStream_t str);
void L2Lkernel_output(cuDoubleComplex* agg, cuDoubleComplex* aggl, int samp, int sampl, int numclus, double* interp, int* intind, int ninter, cuDoubleComplex* shift, dim3 grid, dim3 block, int mem);
void L2Lkernelh_output(cuDoubleComplex* agg, cuDoubleComplex* aggl, int samp, int sampl, int numclus, double* interp, int* intind, int ninter, cuDoubleComplex* shift, dim3 grid, dim3 block, int mem);
void L2Lkernel(cuDoubleComplex* agg, cuDoubleComplex* aggl, int sampl, double* interp, int* intind, int ninter, cuDoubleComplex* shift, dim3 grid, dim3 block, int mem);
void L2Lkernelh(cuDoubleComplex* agg, cuDoubleComplex* aggl, int sampl, double* interp, int* intind, int ninter, cuDoubleComplex* shift, dim3 grid, dim3 block, int mem);
void M2Lkernel_output(cuDoubleComplex* loc, cuDoubleComplex* agg, int samp, int numclus, int* far, int* traid, cuDoubleComplex* trans, dim3 grid, dim3 block, int mem);
void P2Pkernel(cuDoubleComplex* r, cuDoubleComplex* x, int* clusnear, cuDoubleComplex* near, int chunkSize, dim3 grid, dim3 block, int mem, cudaStream_t str);
void locate(cuDoubleComplex* sendbuff, int* sendmap, cuDoubleComplex* aggmulti, int numsamp, dim3 grid, dim3 block, int mem);
void locate(cuDoubleComplex* sendbuff, int* sendmap, cuDoubleComplex* aggmulti, int numsamp, dim3 grid, dim3 block);
void relocate(cuDoubleComplex* aggmulti, int* recvmap, cuDoubleComplex* recvbuff, int numsamp, dim3 grid, dim3 block);
void norm2partc(double* part, cuDoubleComplex* a, dim3 grid, dim3 block, int mem);
void norm2partd(double* part, dim3 grid, dim3 block, int mem);
void innerpartc(cuDoubleComplex* part, cuDoubleComplex* a, cuDoubleComplex* b, dim3 grid, dim3 block, int mem);
void innerpartd(cuDoubleComplex* part, dim3 grid, dim3 block, int mem);
void prep(cuDoubleComplex* buff, cuDoubleComplex* x, cuDoubleComplex* o, dim3 grid, dim3 block);
void prep(cuDoubleComplex* buff, cuDoubleComplex* x, dim3 grid, dim3 block);
void post(cuDoubleComplex* r_d, cuDoubleComplex* x_d, dim3 grid, dim3 block);
void post(cuDoubleComplex* r, cuDoubleComplex* x, cuDoubleComplex* o, dim3 grid, dim3 block);
void saxp(cuDoubleComplex* a, cuDoubleComplex* b, cuDoubleComplex* c, cuDoubleComplex alpha, dim3 grid, dim3 block);


__global__ void P2Pkernel_myModification(cuDoubleComplex* r, cuDoubleComplex* x, int* clusnear, cuDoubleComplex* near);
__global__ void P2Pkernel_myModification_2(cuDoubleComplex* r, cuDoubleComplex* x, int* clusnear, cuDoubleComplex* near);
__global__ void P2Pkernel_redundancy(cuDoubleComplex* r, cuDoubleComplex* x, int* clusnear, cuDoubleComplex* near_redundant, int numRedundantBlocks, int chunkSize);
void P2Pkernel_myModification(cuDoubleComplex* r, cuDoubleComplex* x, int* clusnear, cuDoubleComplex* near, dim3 grid, dim3 block, int mem, cudaStream_t str);
void P2Pkernel_redundancy(cuDoubleComplex* r, cuDoubleComplex* x, int* clusnear, cuDoubleComplex* near_redundant, int numRedundantBlocks, int chunkSize, dim3 grid, dim3 block, int mem, cudaStream_t str);

