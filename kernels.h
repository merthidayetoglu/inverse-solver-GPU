#include "vars.h"

#include <cuda.h>
#include <cuComplex.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void P2Mkernel(cuDoubleComplex*,cuDoubleComplex*,cuDoubleComplex*,int,int,int);
__global__ void L2Pkernel(cuDoubleComplex*,cuDoubleComplex*,cuDoubleComplex*,int,int,int);
__global__ void L2Pkernelh(cuDoubleComplex*,cuDoubleComplex*,cuDoubleComplex*,int,int,int);
__global__ void M2Mkernel_output(cuDoubleComplex*,cuDoubleComplex*,int,int,int,double*,int*,int,cuDoubleComplex*);
__global__ void M2Mkernel(cuDoubleComplex*,cuDoubleComplex*,int,double*,int*,int,cuDoubleComplex*);
__global__ void M2Lkernel_output(cuDoubleComplex*,cuDoubleComplex*,int,int,int*,int*,cuDoubleComplex*);
__global__ void P2Pkernel(cuDoubleComplex*,cuDoubleComplex*,int*,cuDoubleComplex*);
__global__ void L2Lkernel_output(cuDoubleComplex*,cuDoubleComplex*,int,int,int,double*,int*,int,cuDoubleComplex*);
__global__ void L2Lkernelh_output(cuDoubleComplex*,cuDoubleComplex*,int,int,int,double*,int*,int,cuDoubleComplex*);
__global__ void L2Lkernel(cuDoubleComplex*,cuDoubleComplex*,int,double*,int*,int,cuDoubleComplex*);
__global__ void L2Lkernelh(cuDoubleComplex*,cuDoubleComplex*,int,double*,int*,int,cuDoubleComplex*);

__global__ void locate(cuDoubleComplex*,int*,cuDoubleComplex*,int);
__global__ void relocate(cuDoubleComplex*,int*,cuDoubleComplex*,int);

__global__ void norm2partc(double*,cuDoubleComplex*);
__global__ void norm2partd(double*);
__global__ void innerpartc(cuDoubleComplex*,cuDoubleComplex*,cuDoubleComplex*);
__global__ void innerpartd(cuDoubleComplex*);

__global__ void prep(cuDoubleComplex*,cuDoubleComplex*,cuDoubleComplex*);
__global__ void prep(cuDoubleComplex*,cuDoubleComplex*);
__global__ void post(cuDoubleComplex*,cuDoubleComplex*);
__global__ void post(cuDoubleComplex*,cuDoubleComplex*,cuDoubleComplex*);
__global__ void saxp(cuDoubleComplex*,cuDoubleComplex*,cuDoubleComplex*,cuDoubleComplex);

void mlfma_gpu(cuDoubleComplex*,cuDoubleComplex*);
