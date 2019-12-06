#include "kernels.h"

extern int numunk;
extern int level;
extern int box;
extern int ninter;
extern int *numsamp;
extern int *numclus;
extern complex<double> *coeff_multi;
extern complex<double> *basis_local;
extern complex<double> **aggmulti;
extern complex<double> **agglocal;
extern double **interp;
extern double **anterp;
extern int **intind;
extern int **antind;
extern complex<double> **shiftmul;
extern complex<double> **shiftloc;
extern complex<double> **trans;
extern int **traid;
extern int **clusfar;
extern int **clusnear;
extern complex<double> *near;
extern int *unkmap;

extern int **sendmap;
extern int **recvmap;
extern int **procmap;
extern int **clusmap;
extern int **procint;
extern int *clcount;
extern MPI_Comm MPI_COMM_MLFMA;

cuDoubleComplex *coeff_d;
cuDoubleComplex *basis_d;
cuDoubleComplex **aggmulti_d;
cuDoubleComplex **agglocal_d;
double **interp_d;
double **anterp_d;
int **intind_d;
int **antind_d;
cuDoubleComplex **shiftmul_d;
cuDoubleComplex **shiftloc_d;
cuDoubleComplex **trans_d;
int **traid_d;
int **clusfar_d;
int *clusnear_d;
cuDoubleComplex *near_d;

int **sendmap_d;
int **recvmap_d;
int **procmap_d;
int **clusmap_d;
int **procint_d;
int *clcount_d;
cuDoubleComplex **sendbuff_d;
cuDoubleComplex **recvbuff_d;
complex<double> **sendbuff_h;
complex<double> **recvbuff_h;
cudaStream_t commstr;
cudaStream_t kerrstr;

complex<double> *x_h;
complex<double> *r_h;
cuDoubleComplex *x_d;
cuDoubleComplex *r_d;
double gpumem;

void setup_gpu(){
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  cudaStreamCreate(&commstr);
  cudaStreamCreate(&kerrstr);
  complex<double> *coeff_h = new complex<double>[box*box*numsamp[level-1]];
  complex<double> *basis_h = new complex<double>[box*box*numsamp[level-1]];
  #pragma omp parallel for
  for(int m = 0; m < numsamp[level-1]; m++)
    for(int n = 0; n < box*box; n++){
      coeff_h[n*numsamp[level-1]+m] = coeff_multi[m*box*box+n];
      basis_h[m*box*box+n] = basis_local[n*numsamp[level-1]+m];
    }
  complex<double> *near_h = new complex<double>[9*box*box*box*box];
  for(int k = 0; k < 9; k++){
    int start = k*box*box*box*box;
    #pragma omp parallel for
    for(int m = 0; m < box*box; m++)
      for(int n = 0; n < box*box; n++)
        near_h[start+m*box*box+n]=near[start+n*box*box+m];
  }
  int **intind_h = new int*[level];
  double **interp_h = new double*[level];
  for(int i = 0; i < level-1; i++){
    intind_h[i] = new int[numsamp[i]];
    interp_h[i] = new double[numsamp[i]*ninter];
    #pragma omp parallel for
    for(int l = 0; l < numsamp[i]; l++){
      intind_h[i][l] = intind[i][l*ninter];
      for(int n = 0; n < ninter; n++)
        interp_h[i][n*numsamp[i]+l] = interp[i][l*ninter+n];
    }
  }
  int **antind_h = new int*[level];
  double **anterp_h = new double*[level];
  for(int i = 1; i < level; i++){
    antind_h[i] = new int[numsamp[i]];
    anterp_h[i] = new double[numsamp[i]*2*ninter];
    #pragma omp parallel for
    for(int l = 0; l < numsamp[i]; l++){
      antind_h[i][l] = antind[i][l*2*ninter];
      for(int n = 0; n < 2*ninter; n++)
        anterp_h[i][n*numsamp[i]+l] = anterp[i][l*2*ninter+n];
    }
  }
  /*if(myid==0){
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Device Count: %d\n",deviceCount);
  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d name: %d\n",dev,deviceProp.name);
    printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    printf("Maximum global memory size: %d\n",deviceProp.totalGlobalMem);
    printf("Maximum constant memory size: %d\n",deviceProp.totalConstMem);
    printf("Maximum shared memory size per block: %d\n",deviceProp.sharedMemPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Warp size: %d\n",deviceProp.warpSize);
  }
  }*/
  gpumem = 0;
  cudaMallocHost((void**)&x_h,numunk*sizeof(complex<double>));
  cudaMallocHost((void**)&r_h,numunk*sizeof(complex<double>));
  cudaMalloc((void**)&x_d,numunk*sizeof(complex<double>));
  cudaMalloc((void**)&r_d,numunk*sizeof(complex<double>));
  gpumem=gpumem+(double)sizeof(complex<double>)*2/1024/1024;
  cudaMalloc((void**)&coeff_d,numsamp[level-1]*box*box*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&basis_d,numsamp[level-1]*box*box*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&near_d,9*box*box*box*box*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&clusnear_d,numclus[level-1]*9*sizeof(int));
  gpumem=gpumem+(double)sizeof(cuDoubleComplex)*2*numsamp[level-1]*box*box/1024/1024;
  gpumem=gpumem+(double)sizeof(cuDoubleComplex)*9*box*box*box*box/1024/1024;
  gpumem=gpumem+(double)sizeof(int)*9*numclus[level-1]/1024/1024;
  aggmulti_d = new cuDoubleComplex*[level];
  agglocal_d = new cuDoubleComplex*[level];
  interp_d = new double*[level];
  anterp_d = new double*[level];
  intind_d = new int*[level];
  antind_d = new int*[level];
  shiftmul_d = new cuDoubleComplex*[level];
  shiftloc_d = new cuDoubleComplex*[level];
  trans_d = new cuDoubleComplex*[level];
  traid_d = new int*[level];
  clusfar_d = new int*[level];
  gpumem=gpumem+(double)sizeof(cuDoubleComplex*)*5*level/1024/1024;
  gpumem=gpumem+(double)sizeof(double*)*2*level/1024/1024;
  gpumem=gpumem+(double)sizeof(int*)*4*level/1024/1024;

  sendmap_d = new int*[level+1];
  recvmap_d = new int*[level+1];
  clcount_d = new int[level+1];
  sendbuff_d = new cuDoubleComplex*[level+1];
  recvbuff_d = new cuDoubleComplex*[level+1];
  sendbuff_h = new complex<double>*[level+1];
  recvbuff_h = new complex<double>*[level+1];
  gpumem=gpumem+(double)sizeof(int*)*2*(level+1)/1024/1024;
  gpumem=gpumem+(double)sizeof(cuDoubleComplex*)*2*(level+1)/1024/1024;
  gpumem=gpumem+(double)sizeof(int)*(level+1)/1024/1024;
  for(int i = 2; i < level+1; i++){
    cudaMalloc((void**)&sendmap_d[i],clcount[i]*sizeof(int));
    cudaMalloc((void**)&recvmap_d[i],clcount[i]*sizeof(int));
    gpumem=gpumem+(double)sizeof(int)*2*clcount[i]/1024/1024;
    if(i < level){
      cudaMalloc((void**)&sendbuff_d[i],clcount[i]*numsamp[i]*sizeof(cuDoubleComplex));
      cudaMalloc((void**)&recvbuff_d[i],clcount[i]*numsamp[i]*sizeof(cuDoubleComplex));
      cudaMallocHost((void**)&sendbuff_h[i],clcount[i]*numsamp[i]*sizeof(complex<double>));
      cudaMallocHost((void**)&recvbuff_h[i],clcount[i]*numsamp[i]*sizeof(complex<double>));
      gpumem=gpumem+(double)sizeof(cuDoubleComplex)*2*clcount[i]*numsamp[i]/1024/1024;
    }else{
      cudaMalloc((void**)&sendbuff_d[i],clcount[i]*box*box*sizeof(cuDoubleComplex));
      cudaMalloc((void**)&recvbuff_d[i],clcount[i]*box*box*sizeof(cuDoubleComplex));
      cudaMallocHost((void**)&sendbuff_h[i],clcount[i]*box*box*sizeof(complex<double>));
      cudaMallocHost((void**)&recvbuff_h[i],clcount[i]*box*box*sizeof(complex<double>));
      gpumem=gpumem+(double)sizeof(cuDoubleComplex)*2*clcount[i]*box*box/1024/1024;
    }
  }
  for(int i = 0; i < level; i++){
    cudaMalloc((void**)&aggmulti_d[i],numclus[i]*numsamp[i]*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&agglocal_d[i],numclus[i]*numsamp[i]*sizeof(cuDoubleComplex));
    gpumem=gpumem+(double)sizeof(cuDoubleComplex)*2*numclus[i]*numsamp[i]/1024/1024;
    if(i > 1){
      cudaMalloc((void**)&trans_d[i],49*numsamp[i]*sizeof(cuDoubleComplex));
      cudaMalloc((void**)&traid_d[i],27*numclus[i]*sizeof(int));
      cudaMalloc((void**)&clusfar_d[i],27*numclus[i]*sizeof(int));
      gpumem=gpumem+(double)sizeof(cuDoubleComplex)*49*numsamp[i]/1024/1024;
      gpumem=gpumem+(double)sizeof(int)*2*27*numclus[i]/1024/1024;
    }
  }
  for(int i = 0; i < level-1; i++){
    cudaMalloc((void**)&interp_d[i],numsamp[i]*ninter*sizeof(double));
    cudaMalloc((void**)&anterp_d[i+1],2*numsamp[i+1]*ninter*sizeof(double));
    cudaMalloc((void**)&intind_d[i],numsamp[i]*sizeof(int));
    cudaMalloc((void**)&antind_d[i+1],numsamp[i+1]*sizeof(int));
    cudaMalloc((void**)&shiftmul_d[i],4*numsamp[i]*sizeof(cuDoubleComplex));
    cudaMalloc((void**)&shiftloc_d[i],4*numsamp[i]*sizeof(cuDoubleComplex));
    gpumem=gpumem+(double)sizeof(double)*numsamp[i]*ninter/1024/1024;
    gpumem=gpumem+(double)sizeof(double)*2*numsamp[i+1]*ninter/1024/1024;
    gpumem=gpumem+(double)sizeof(int)*numsamp[i]/1024/1024;
    gpumem=gpumem+(double)sizeof(int)*2*numsamp[i+1]/1024/1024;
    gpumem=gpumem+(double)sizeof(cuDoubleComplex)*2*4*numsamp[i]/1024/1024;
  }
  if(myid==0)printf("GPU MEMORY: %f MB\n",gpumem);
  int memory;
  memory = (level+1)*sizeof(int);
  cudaMemcpy(clcount_d,clcount,memory*sizeof(int),cudaMemcpyHostToDevice);
  for(int i = 2; i < level+1; i++){
    memory = clcount[i]*sizeof(int);
    cudaMemcpy(sendmap_d[i],sendmap[i],memory,cudaMemcpyHostToDevice);
    cudaMemcpy(recvmap_d[i],recvmap[i],memory,cudaMemcpyHostToDevice);
  }
  memory = numsamp[level-1]*box*box*sizeof(cuDoubleComplex);
  cudaMemcpy(coeff_d,coeff_h,memory,cudaMemcpyHostToDevice);
  cudaMemcpy(basis_d,basis_h,memory,cudaMemcpyHostToDevice);
  memory = numclus[level-1]*9*sizeof(int);
  cudaMemcpy(clusnear_d,clusnear[level-1],memory,cudaMemcpyHostToDevice);
  memory = 9*box*box*box*box*sizeof(cuDoubleComplex);
  cudaMemcpy(near_d,near_h,memory,cudaMemcpyHostToDevice);
  for(int i = 2; i < level-1; i++){
    memory = numsamp[i]*ninter*sizeof(double);
    cudaMemcpy(interp_d[i],interp_h[i],memory,cudaMemcpyHostToDevice);
    memory = 2*numsamp[i+1]*ninter*sizeof(double);
    cudaMemcpy(anterp_d[i+1],anterp_h[i+1],memory,cudaMemcpyHostToDevice);
    memory = numsamp[i]*sizeof(int);
    cudaMemcpy(intind_d[i],intind_h[i],memory,cudaMemcpyHostToDevice);
    memory = numsamp[i+1]*sizeof(int);
    cudaMemcpy(antind_d[i+1],antind_h[i+1],memory,cudaMemcpyHostToDevice);
    memory = 4*numsamp[i]*sizeof(cuDoubleComplex);
    cudaMemcpy(shiftmul_d[i],shiftmul[i],memory,cudaMemcpyHostToDevice);
    cudaMemcpy(shiftloc_d[i],shiftloc[i],memory,cudaMemcpyHostToDevice);
  }
  for(int i = 2; i < level; i++){
    memory = 49*numsamp[i]*sizeof(cuDoubleComplex);
    cudaMemcpy(trans_d[i],trans[i],memory,cudaMemcpyHostToDevice);
    memory = 27*numclus[i]*sizeof(int);
    cudaMemcpy(traid_d[i],traid[i],memory,cudaMemcpyHostToDevice);
    cudaMemcpy(clusfar_d[i],clusfar[i],memory,cudaMemcpyHostToDevice);
  }
  delete[] coeff_h;
  delete[] basis_h;
  delete[] near_h;
  for(int i = 0; i < level-1; i++){
    delete[] intind_h[i];
    delete[] antind_h[i+1];
  }
  delete[] intind_h;
  delete[] antind_h;
}
