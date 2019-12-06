#include "kernels.h"

  extern int ninter;
  extern int level;
  extern int box;
  extern int numunk;
  extern int *numclus;
  extern int *numsamp;
  extern cuDoubleComplex *coeff_d;
  extern cuDoubleComplex *basis_d;
  extern cuDoubleComplex **aggmulti_d;
  extern cuDoubleComplex **agglocal_d;
  extern double **interp_d;
  extern double **anterp_d;
  extern int **intind_d;
  extern int **antind_d;
  extern cuDoubleComplex **shiftmul_d;
  extern cuDoubleComplex **shiftloc_d;
  extern cuDoubleComplex **trans_d;
  extern int **traid_d;
  extern int **clusfar_d;
  extern int *clusnear_d;
  extern cuDoubleComplex *near_d;

  extern int **sendmap;
  extern int **recvmap;
  extern int **procmap;
  extern int **clusmap;
  extern int **procint;
  extern int *clcount;
  extern complex<double> **sendbuff;
  extern complex<double> **recvbuff;

  extern int **sendmap_d;
  extern int **recvmap_d;
  extern int **procmap_d;
  extern int **clusmap_d;
  extern int **procint_d;
  extern int *clcount_d;
  extern cuDoubleComplex **sendbuff_d;
  extern cuDoubleComplex **recvbuff_d;
  extern cudaStream_t commstr;
  extern cudaStream_t kerrstr;

  extern cuDoubleComplex *x_d;
  extern cuDoubleComplex *r_d;

  extern complex<double> **aggmulti;
  extern complex<double> **agglocal;
  extern double **interp;
  extern double **anterp;
  extern int **intind;
  extern int **antind;
  extern complex<double> **shiftmul;
  extern complex<double> **shiftloc;
  extern complex<double> **temp;

  extern int matvecin;
  extern int matvecout;
  extern double innert;
  extern double outert;
  extern MPI_Comm MPI_COMM_MLFMA;
  extern MPI_Comm MPI_COMM_DBIM;

void mlfma(complex<double> *x, complex<double> *r){
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  int memory;
  int addres;

  //TRANSFER DATA IN
  memory = numunk/numproc*sizeof(cuDoubleComplex);
  addres = myid*numunk/numproc;
  cudaMemcpy(&x_d[addres],&x[addres],memory,cudaMemcpyHostToDevice);
  mlfma_gpu(x_d,r_d);
  cudaMemcpy(&r[addres],&r_d[addres],memory,cudaMemcpyDeviceToHost);
}
void mlfmah(complex<double> *x, complex<double> *r){
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  int memory;
  int addres;
  #pragma omp parallel for
  for(int n = myid*numunk/numproc; n < (myid+1)*numunk/numproc; n++)
    x[n] = conj(x[n]);
  //TRANSFER DATA IN
  memory = numunk/numproc*sizeof(cuDoubleComplex);
  addres = myid*numunk/numproc;
  cudaMemcpy(&x_d[addres],&x[addres],memory,cudaMemcpyHostToDevice);
  mlfma_gpu(x_d,r_d);
  cudaMemcpy(&r[addres],&r_d[addres],memory,cudaMemcpyDeviceToHost);
  #pragma omp parallel for
  for(int n = myid*numunk/numproc; n < (myid+1)*numunk/numproc; n++){
    r[n] = conj(r[n]);
    x[n] = conj(x[n]);
  }
}

void mlfma_gpu(cuDoubleComplex *x_d, cuDoubleComplex *r_d){

  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);

  float miliseconds;
  int memory;
  int addres;
  int addrss;
  int tile;
  int tilex;
  int tiley;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  matvecin++;
  cudaEventRecord(start);

  if(clcount[level])locate<<<clcount[level],box*box>>>(sendbuff_d[level],sendmap_d[level],x_d,box*box);

  //LOWEST-LEVEL AGGREGATION
  tile = 16;
  memory = 2*tile*tile*sizeof(cuDoubleComplex);
  addres = myid*numunk/numproc;
  addrss = myid*numclus[level-1]/numproc*numsamp[level-1];
  dim3 P2Mgrid(numsamp[level-1]/tile,(numclus[level-1]/numproc)/tile,1);
  if(numsamp[level-1]%tile)P2Mgrid.x++;
  if((numclus[level-1]/numproc)%tile)P2Mgrid.y++;
  dim3 P2Mblock(tile,tile,1);
  P2Mkernel<<<P2Mgrid,P2Mblock,memory,kerrstr>>>(&aggmulti_d[level-1][addrss],&x_d[addres],coeff_d,numclus[level-1],box*box,numsamp[level-1]);
  //HIGHER-LEVEL AGGREGATION
  for(int i = level-2; i > 1; i--){
    if(numsamp[i] < 1024){
      memory = numsamp[i+1]*4*sizeof(cuDoubleComplex);
      addres = myid*numclus[i]/numproc*numsamp[i];
      addrss = myid*numclus[i+1]/numproc*numsamp[i+1];
      //dim3 M2Mgrid(sqrt(numclus[i]/numproc),sqrt(numclus[i]/numproc),1);
      dim3 M2Mgrid(numclus[i]/numproc,1,1);
      dim3 M2Mblock(numsamp[i],1,1);
      M2Mkernel<<<M2Mgrid,M2Mblock,memory,kerrstr>>>(&aggmulti_d[i][addres],&aggmulti_d[i+1][addrss],numsamp[i+1],interp_d[i],intind_d[i],ninter,shiftmul_d[i]);
    }else{
      tilex = 16;
      tiley = 16;
      memory = (ninter+4)*tilex*sizeof(cuDoubleComplex)+tilex*sizeof(int);
      addres = myid*numclus[i]/numproc*numsamp[i];
      addrss = myid*numclus[i+1]/numproc*numsamp[i+1];
      dim3 M2Mgrid(numsamp[i]/tilex,(numclus[i]/numproc)/tiley,1);
      if(numsamp[i]%tilex)M2Mgrid.x++;
      if((numclus[i]/numproc)%tiley)M2Mgrid.y++;
      dim3 M2Mblock(tilex,tiley,1);
      M2Mkernel_output<<<M2Mgrid,M2Mblock,memory,kerrstr>>>(&aggmulti_d[i][addres],&aggmulti_d[i+1][addrss],numsamp[i],numsamp[i+1],numclus[i]/numproc,interp_d[i],intind_d[i],ninter,shiftmul_d[i]);
    }
  }
  //NEARFIELD COMMUNICATION
  if(clcount[level]){
    memory = clcount[level]*box*box*sizeof(cuDoubleComplex);
    cudaMemcpyAsync(sendbuff[level],sendbuff_d[level],memory,cudaMemcpyDeviceToHost,commstr);
    cudaStreamSynchronize(commstr);
    for(int p = 0; p < numproc; p++){
      int sib = procmap[level][p];
      if(sib != -1){
        complex<double> *send = &sendbuff[level][procint[level][p]*box*box];
        complex<double> *recv = &recvbuff[level][procint[level][p]*box*box];
        int amount = clusmap[level][p]*box*box;
        MPI_Sendrecv(send,amount,MPI_DOUBLE_COMPLEX,sib,0,recv,amount,MPI_DOUBLE_COMPLEX,sib,0,MPI_COMM_MLFMA,MPI_STATUS_IGNORE);
      }
    }
    memory = clcount[level]*box*box*sizeof(cuDoubleComplex);
    cudaMemcpyAsync(recvbuff_d[level],recvbuff[level],memory,cudaMemcpyHostToDevice,commstr);
    relocate<<<clcount[level],box*box>>>(x_d,recvmap_d[level],recvbuff_d[level],box*box);
  }
  for(int i = 2; i < level && clcount[i]; i++)
    locate<<<clcount[i],128,0>>>(sendbuff_d[i],sendmap_d[i],aggmulti_d[i],numsamp[i]);

  //NEARFIELD MULTIPLICATION
  memory = 4*9*sizeof(int)+8*box*box*sizeof(cuDoubleComplex);
  addres = myid*numunk/numproc;
  addrss = myid*numclus[level-1]/numproc*9;
  dim3 P2Pgrid(numclus[level-1]/numproc/4,1,1);
  dim3 P2Pblock(4*box*box,1,1);
  P2Pkernel<<<P2Pgrid,P2Pblock,memory,kerrstr>>>(&r_d[addres],x_d,&clusnear_d[addrss],near_d);

  //FARFIELD COMMUNICATION
  for(int i = 2; i < level && clcount[i]; i++){
    memory=clcount[i]*numsamp[i]*sizeof(cuDoubleComplex);
    cudaMemcpyAsync(sendbuff[i],sendbuff_d[i],memory,cudaMemcpyDeviceToHost,commstr);
  }
  cudaStreamSynchronize(commstr);
  for(int i = 2; i < level; i++){
    memory = clcount[i]*numsamp[i]*sizeof(cuDoubleComplex);
    for(int p = 0; p < numproc; p++){
      int sib = procmap[i][p];
      if(sib != -1){
        complex<double> *send = &sendbuff[i][procint[i][p]*numsamp[i]];
        complex<double> *recv = &recvbuff[i][procint[i][p]*numsamp[i]];
        int amount = clusmap[i][p]*numsamp[i];
        MPI_Sendrecv(send,amount,MPI_DOUBLE_COMPLEX,sib,0,recv,amount,MPI_DOUBLE_COMPLEX,sib,0,MPI_COMM_MLFMA,MPI_STATUS_IGNORE);
      }
    }
    memory=clcount[i]*numsamp[i]*sizeof(cuDoubleComplex);
    cudaMemcpyAsync(recvbuff_d[i],recvbuff[i],memory,cudaMemcpyHostToDevice,commstr);
  }
  for(int i = 2; i < level && clcount[i]; i++)
    relocate<<<clcount[i],128,0>>>(aggmulti_d[i],recvmap_d[i],recvbuff_d[i],numsamp[i]);

  //TRANSLATION
  for(int i = level-1; i > 1; i--){
    tilex = 8;
    tiley = 16;
    memory = 2*tiley*27*sizeof(int);
    addres = myid*numclus[i]/numproc*numsamp[i];
    addrss = myid*numclus[i]/numproc*27;
    dim3 M2Lgrid(numsamp[i]/tilex,(numclus[i]/numproc)/tiley,1);
    if(numsamp[i]%tilex)M2Lgrid.x++;
    if((numclus[i]/numproc)%tiley)M2Lgrid.y++;
    dim3 M2Lblock(tilex,tiley,1);
    M2Lkernel_output<<<M2Lgrid,M2Lblock,memory>>>(&agglocal_d[i][addres],aggmulti_d[i],numsamp[i],numclus[i]/numproc,&clusfar_d[i][addrss],&traid_d[i][addrss],trans_d[i]);
  }
  //HIGHER-LEVEL DISAGGREGATION
  for(int i = 2; i < level-1; i++){
    if(numsamp[i] > 512){
      tilex = 4;
      tiley = 16;
      memory = 2*ninter*tilex*sizeof(cuDoubleComplex)+tilex*sizeof(int);
      addres = myid*numclus[i]/numproc*numsamp[i];
      addrss = myid*numclus[i+1]/numproc*numsamp[i+1];
      dim3 L2Lgrid(numsamp[i+1]/tilex,numclus[i]/tiley,1);
      if(numsamp[i+1]%tilex)L2Lgrid.x++;
      if(numclus[i]%tiley)L2Lgrid.y++;
      dim3 L2Lblock(tilex,tiley,1);
      L2Lkernel_output<<<L2Lgrid,L2Lblock,memory>>>(&agglocal_d[i][addres],&agglocal_d[i+1][addrss],numsamp[i],numsamp[i+1],numclus[i]/numproc,anterp_d[i+1],antind_d[i+1],2*ninter,shiftloc_d[i]);
    }else{
      memory = 4*numsamp[i]*sizeof(cuDoubleComplex);
      addres = myid*numclus[i]/numproc*numsamp[i];
      addrss = myid*numclus[i+1]/numproc*numsamp[i+1];
      //dim3 L2Lgrid(sqrt(numclus[i]/numproc),sqrt(numclus[i]/numproc),1);
      dim3 L2Lgrid(numclus[i]/numproc,1,1);
      dim3 L2Lblock(numsamp[i],1,1);
      L2Lkernel<<<L2Lgrid,L2Lblock,memory>>>(&agglocal_d[i][addres],&agglocal_d[i+1][addrss],numsamp[i+1],anterp_d[i+1],antind_d[i+1],2*ninter,shiftloc_d[i]);
    }
  }
  //LOWEST-LEVEL DISAGGREGATION
  tile = 16;
  memory = 2*tile*tile*sizeof(cuDoubleComplex);
  addres = myid*numunk/numproc;
  addrss = myid*numclus[level-1]/numproc*numsamp[level-1];
  dim3 L2Pgrid((box*box)/tile,(numclus[level-1]/numproc)/tile,1);
  if((box*box)%tile)L2Pgrid.x++;
  if((numclus[level-1]/numproc)%tile)L2Pgrid.y++;
  dim3 L2Pblock(tile,tile,1);
  L2Pkernel<<<L2Pgrid,L2Pblock,memory>>>(&r_d[addres],&agglocal_d[level-1][addrss],basis_d,numclus[level-1],numsamp[level-1],box*box);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&miliseconds,start,stop);
  innert = innert + miliseconds/1000;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void aggregate(complex<double> *x, complex<double> *r){
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);

  float miliseconds;
  int memory;
  int addres;
  int addrss;
  int tile;
  int tilex;
  int tiley;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  matvecout++;
  cudaEventRecord(start);

  //TRANSFER DATA IN
  memory = numunk/numproc*sizeof(cuDoubleComplex);
  addres = myid*numunk/numproc;
  cudaMemcpy(&x_d[addres],&x[addres],memory,cudaMemcpyHostToDevice);

  //LOWEST-LEVEL AGGREGATION
  tile = 16;
  memory = 2*tile*tile*sizeof(cuDoubleComplex);
  addres = myid*numunk/numproc;
  addrss = myid*numclus[level-1]/numproc*numsamp[level-1];
  dim3 P2Mgrid(numsamp[level-1]/tile,(numclus[level-1]/numproc)/tile,1);
  if(numsamp[level-1]%tile)P2Mgrid.x++;
  if((numclus[level-1]/numproc)%tile)P2Mgrid.y++;
  dim3 P2Mblock(tile,tile,1);
  P2Mkernel<<<P2Mgrid,P2Mblock,memory,kerrstr>>>(&aggmulti_d[level-1][addrss],&x_d[addres],coeff_d,numclus[level-1],box*box,numsamp[level-1]);
  //HIGHER-LEVEL AGGREGATION
  for(int i = level-2; i > 1; i--){
    if(numsamp[i] < 1024){
      memory = numsamp[i+1]*4*sizeof(cuDoubleComplex);
      addres = myid*numclus[i]/numproc*numsamp[i];
      addrss = myid*numclus[i+1]/numproc*numsamp[i+1];
      //dim3 M2Mgrid(sqrt(numclus[i]/numproc),sqrt(numclus[i]/numproc),1);
      dim3 M2Mgrid(numclus[i]/numproc,1,1);
      dim3 M2Mblock(numsamp[i],1,1);
      M2Mkernel<<<M2Mgrid,M2Mblock,memory,kerrstr>>>(&aggmulti_d[i][addres],&aggmulti_d[i+1][addrss],numsamp[i+1],interp_d[i],intind_d[i],ninter,shiftmul_d[i]);
    }else{
      tilex = 16;
      tiley = 16;
      memory = (ninter+4)*tilex*sizeof(cuDoubleComplex)+tilex*sizeof(int);
      addres = myid*numclus[i]/numproc*numsamp[i];
      addrss = myid*numclus[i+1]/numproc*numsamp[i+1];
      dim3 M2Mgrid(numsamp[i]/tilex,(numclus[i]/numproc)/tiley,1);
      if(numsamp[i]%tilex)M2Mgrid.x++;
      if((numclus[i]/numproc)%tiley)M2Mgrid.y++;
      dim3 M2Mblock(tilex,tiley,1);
      M2Mkernel_output<<<M2Mgrid,M2Mblock,memory,kerrstr>>>(&aggmulti_d[i][addres],&aggmulti_d[i+1][addrss],numsamp[i],numsamp[i+1],numclus[i]/numproc,interp_d[i],intind_d[i],ninter,shiftmul_d[i]);
    }
  }
  //TRANSFER DATA OUT
  int amount = numclus[2]*numsamp[2]/numproc;
  addres = myid*amount;
  memory = amount*sizeof(cuDoubleComplex);
  cudaMemcpy(&aggmulti[2][addres],&aggmulti_d[2][addres],memory,cudaMemcpyDeviceToHost);
  MPI_Allgather(&aggmulti[2][addres],amount,MPI_DOUBLE_COMPLEX,agglocal[2],amount,MPI_DOUBLE_COMPLEX,MPI_COMM_MLFMA);
  memcpy(aggmulti[2],agglocal[2],numclus[2]*numsamp[2]*sizeof(complex<double>));
  //TOP-LEVEL AGGREGATIONS
  for(int i = 1; i > -1; i--){
    #pragma omp parallel for
    for(int clusm = 0; clusm < numclus[i]; clusm++){
      int indm = clusm*numsamp[i];
      for(int km = 0; km < numsamp[i]; km++){
        complex<double> temp1 = 0;
        for(int cn = 0; cn < 4; cn++){
          int indn = (clusm*4+cn)*numsamp[i+1];
          complex<double> reduce = 0;
          //INTERPOLATE
          for(int k = 0; k < ninter; k++)
            reduce = reduce+interp[i][km*ninter+k]*aggmulti[i+1][indn+intind[i][km*ninter+k]];
          //SHIFT
          temp1 = temp1 + reduce*shiftmul[i][cn*numsamp[i]+km];
        }
        aggmulti[i][indm+km]=temp1;
      }
    }
  }
  extern double *interph;
  extern int *intindh;
  extern int numrx;
  //TOP-LEVEL INTERPOLATION
  #pragma omp parallel for
  for(int m = 0; m < numrx; m++){
    complex<double> reduce = 0;
    for(int k = 0; k < ninter; k++)
      reduce = reduce+interph[m*ninter+k]*aggmulti[0][intindh[m*ninter+k]];
    r[m] = reduce;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&miliseconds,start,stop);
  outert = outert + miliseconds/1000;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
void aggregateh(complex<double> *x, complex<double> *r){
  int myid;
  int numproc;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);

  float miliseconds;
  int memory;
  int addres;
  int addrss;
  int tile;
  int tilex;
  int tiley;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  matvecout++;
  cudaEventRecord(start);

  extern double *anterph;
  extern int *antindh;
  extern double res;
  extern int numrx;
  double timet = MPI_Wtime();
  matvecout++;
  #pragma omp parallel for
  for(int m = 0; m < numsamp[0]; m++){
    complex<double> reduce = 0;
    for(int k = 0; k < 2*ninter; k++)
      reduce = reduce+anterph[m*2*ninter+k]*r[antindh[m*2*ninter+k]];
    agglocal[0][m] = reduce*complex<double>(0,-4)*complex<double>(numrx*res*res);
  }
  //TOP-LEVEL DISAGGREGATIONS
  for(int i = 0; i < 2; i++){
    #pragma omp parallel for
    for(int clusn = 0; clusn < numclus[i]; clusn++){
      int indn = clusn*numsamp[i];
      int mythread = omp_get_thread_num();
      for(int cm = 0; cm < 4; cm++){
        int indm = (clusn*4+cm)*numsamp[i+1];
        //SHIFT
        for(int kn = 0; kn < numsamp[i]; kn++)
          temp[mythread][kn] = shiftloc[i][cm*numsamp[i]+kn]*agglocal[i][indn+kn];
        //ANTERPOLATE
        for(int km = 0; km < numsamp[i+1]; km++){
          complex<double> reduce = 0;
          for(int k = 0; k < 2*ninter; k++)
            reduce=reduce+anterp[i+1][km*2*ninter+k]*temp[mythread][antind[i+1][km*2*ninter+k]];
          agglocal[i+1][indm+km]=reduce;
        }
      }
    }
  }
  //TRANSFER DATA IN
  int amount = numclus[2]*numsamp[2]/numproc;
  addres = myid*amount;
  memory = amount*sizeof(cuDoubleComplex);
  cudaMemcpy(&agglocal_d[2][addres],&agglocal[2][addres],memory,cudaMemcpyHostToDevice);
  //HIGHER-LEVEL DISAGGREGATION
  for(int i = 2; i < level-1; i++){
    if(numsamp[i] > 512){
      tilex = 4;
      tiley = 16;
      memory = 2*ninter*tilex*sizeof(cuDoubleComplex)+tilex*sizeof(int);
      addres = myid*numclus[i]/numproc*numsamp[i];
      addrss = myid*numclus[i+1]/numproc*numsamp[i+1];
      dim3 L2Lgrid(numsamp[i+1]/tilex,numclus[i]/tiley,1);
      if(numsamp[i+1]%tilex)L2Lgrid.x++;
      if(numclus[i]%tiley)L2Lgrid.y++;
      dim3 L2Lblock(tilex,tiley,1);
      L2Lkernelh_output<<<L2Lgrid,L2Lblock,memory>>>(&agglocal_d[i][addres],&agglocal_d[i+1][addrss],numsamp[i],numsamp[i+1],numclus[i]/numproc,anterp_d[i+1],antind_d[i+1],2*ninter,shiftloc_d[i]);
    }else{
      memory = 4*numsamp[i]*sizeof(cuDoubleComplex);
      addres = myid*numclus[i]/numproc*numsamp[i];
      addrss = myid*numclus[i+1]/numproc*numsamp[i+1];
      //dim3 L2Lgrid(sqrt(numclus[i]/numproc),sqrt(numclus[i]/numproc),1);
      dim3 L2Lgrid(numclus[i]/numproc,1,1);
      dim3 L2Lblock(numsamp[i],1,1);
      L2Lkernelh<<<L2Lgrid,L2Lblock,memory>>>(&agglocal_d[i][addres],&agglocal_d[i+1][addrss],numsamp[i+1],anterp_d[i+1],antind_d[i+1],2*ninter,shiftloc_d[i]);
    }
  }
  //LOWEST-LEVEL DISAGGREGATION
  tile = 16;
  memory = 2*tile*tile*sizeof(cuDoubleComplex);
  addres = myid*numunk/numproc;
  addrss = myid*numclus[level-1]/numproc*numsamp[level-1];
  dim3 L2Pgrid((box*box)/tile,(numclus[level-1]/numproc)/tile,1);
  if((box*box)%tile)L2Pgrid.x++;
  if((numclus[level-1]/numproc)%tile)L2Pgrid.y++;
  dim3 L2Pblock(tile,tile,1);
  L2Pkernelh<<<L2Pgrid,L2Pblock,memory>>>(&r_d[addres],&agglocal_d[level-1][addrss],basis_d,numclus[level-1],numsamp[level-1],box*box);
  //TRANSFER DATA OUT
  memory = numunk/numproc*sizeof(cuDoubleComplex);
  cudaMemcpy(&x[addres],&r_d[addres],memory,cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&miliseconds,start,stop);
  outert = outert + miliseconds/1000;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
