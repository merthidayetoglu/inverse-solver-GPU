#include "kernels.h"

extern int numunk;
extern int iti;
extern double toli;
extern double mem;

cuDoubleComplex *cbuff;
complex<double> *cbuff_h;
double *dbuff;
double *dbuff_h;

extern MPI_Comm MPI_COMM_MLFMA;
extern double bicgst;

double norm2(cuDoubleComplex *a){
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  int addres = myid*numunk/numproc;
  int blocksize = 128;
  int memory = 2*blocksize*sizeof(double);
  int numblocks = numunk/numproc/(2*blocksize);
  norm2partc<<<numblocks,blocksize,memory>>>(&dbuff[addres],&a[addres]);
  while(numblocks/(2*blocksize) > 0){
    numblocks = numblocks/(2*blocksize);
    norm2partd<<<numblocks,blocksize,memory>>>(&dbuff[addres]);
  }
  cudaMemcpy(&dbuff_h[addres],&dbuff[addres],numblocks*sizeof(double),cudaMemcpyDeviceToHost);
  double reduce = 0;
  for(int m = 0; m < numblocks; m++)
    reduce = reduce + dbuff_h[addres+m];
  double rettot;
  MPI_Allreduce(&reduce,&rettot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_MLFMA);
  return rettot;
}
complex<double> inner(cuDoubleComplex *a, cuDoubleComplex *b){
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  int addres = myid*numunk/numproc;
  int blocksize = 128;
  int memory = 2*blocksize*sizeof(cuDoubleComplex);
  int numblocks = numunk/numproc/(2*blocksize);
  innerpartc<<<numblocks,blocksize,memory>>>(&cbuff[addres],&a[addres],&b[addres]);
  while(numblocks/(2*blocksize) > 0){
    numblocks = numblocks/(2*blocksize);
    innerpartd<<<numblocks,blocksize,memory>>>(&cbuff[addres]);
  }
  cudaMemcpy(&cbuff_h[addres],&cbuff[addres],numblocks*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
  complex<double> reduce = 0;
  for(int m = 0; m < numblocks; m++)
    reduce = reduce + cbuff_h[addres+m];
  complex<double> rettot;
  MPI_Allreduce(&reduce,&rettot,1,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_MLFMA);
  return rettot;
}

void matvec(cuDoubleComplex *x, cuDoubleComplex *o, cuDoubleComplex *r, bool ishermitian){
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid); 
  int addres = myid*numunk/numproc;
  if(ishermitian){
    prep<<<numunk/numproc/256,256>>>(&cbuff[addres],&x[addres]);
    mlfma_gpu(cbuff,r);
    post<<<numunk/numproc/256,256>>>(&r[addres],&x[addres],&o[addres]);
  }else{
    prep<<<numunk/numproc/256,256>>>(&cbuff[addres],&x[addres],&o[addres]);
    mlfma_gpu(cbuff,r);
    post<<<numunk/numproc/256,256>>>(&r[addres],&x[addres]);
  }
}

cuDoubleComplex *p;
cuDoubleComplex *v;
cuDoubleComplex *s;
cuDoubleComplex *t;
cuDoubleComplex *r_tld;

cuDoubleComplex *o_d;
cuDoubleComplex *b_d;
extern cuDoubleComplex *x_d;
extern cuDoubleComplex *r_d;

void setup_bicgs(){

  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  cudaMalloc((void**)&p,numunk*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&v,numunk*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&s,numunk*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&t,numunk*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&r_tld,numunk*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&cbuff,numunk*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&dbuff,numunk*sizeof(double));
  cudaMallocHost((void**)&cbuff_h,numunk*sizeof(cuDoubleComplex));
  cudaMallocHost((void**)&dbuff_h,numunk*sizeof(double));
  cudaMalloc((void**)&o_d,numunk*sizeof(cuDoubleComplex));
  cudaMalloc((void**)&b_d,numunk*sizeof(cuDoubleComplex));

  extern double gpumem;
  double memtemp = 0;
  memtemp = memtemp + (double)sizeof(cuDoubleComplex)*10*numunk/1024.0/1024.0;
  memtemp = memtemp + (double)sizeof(double)*numunk/1024.0/1024.0;
  if(myid==0)printf("GPU BICGSTAB SOLVER MEM: %f MB\n",memtemp);
  if(myid==0)printf("GPU TOTAL MEM: %f MB\n",gpumem+memtemp);
}

void bicgs(complex<double> *x, complex<double> *o, complex<double> *b, bool ishermitian){

  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  int numproc_world;
  int myid_world;
  MPI_Comm_size(MPI_COMM_WORLD,&numproc_world);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid_world);
  double timet = MPI_Wtime();

  int addres =  myid*numunk/numproc;
  cudaMemcpy(&o_d[addres],&o[addres],numunk/numproc*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
  cudaMemcpy(&b_d[addres],&b[addres],numunk/numproc*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
  cudaMemcpy(&x_d[addres],&x[addres],numunk/numproc*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);

  int max_it = iti;
  double res_tol = toli;
  complex<double> alpha;
  complex<double> beta;
  complex<double> omega;
  complex<double> rho;
  complex<double> rho_1;

  int iter = 0;
  int nummatvec = 0;
  double rnrm2;
  double snrm2;
  double bnrm2;
  double error;

  bnrm2 = sqrt(norm2(b_d));
  if(bnrm2 < 1e-20)
    bnrm2 = 1;
  //MATVEC 0 START
  matvec(x_d,o_d,r_d,ishermitian);
  nummatvec++;
  //MATVEC 0 FINISH
  saxp<<<numunk/numproc/256,256>>>(&b_d[addres],&r_d[addres],&r_d[addres],make_cuDoubleComplex(-1,0));
  rnrm2 = sqrt(norm2(r_d));
  error = rnrm2/bnrm2;
  //if(myid_world==0)printf("RES. ERROR: %e ITER: %d\n",error,iter);
  if(error > res_tol){
    saxp<<<numunk/numproc/256,256>>>(&r_d[addres],&r_d[addres],&r_tld[addres],make_cuDoubleComplex(0,0));
    //BEGIN ITERATIONS
    while(iter < max_it){
      iter++;
      rho = inner(r_tld,r_d);
      if(abs(rho) < 1e-200){
        printf("RHO BREAKDOWN\n");
        break;
      }
      if(iter == 1)
        saxp<<<numunk/numproc/256,256>>>(&r_d[addres],&r_d[addres],&p[addres],make_cuDoubleComplex(0,0));
      else{
        beta = (rho/rho_1)*(alpha/omega);
        double omegar = omega.real();
        double omegai = omega.imag();
        saxp<<<numunk/numproc/256,256>>>(&p[addres],&v[addres],&cbuff[addres],make_cuDoubleComplex(-1*omegar,-1*omegai));
        double betar = beta.real();
        double betai = beta.imag();
        saxp<<<numunk/numproc/256,256>>>(&r_d[addres],&cbuff[addres],&p[addres],make_cuDoubleComplex(betar,betai));
      }
      //PRECONDITIONER
      //MATVEC 1 START
      matvec(p,o_d,v,ishermitian);
      nummatvec++;
      //MATVEC 1 FINISH
      alpha = rho/inner(r_tld,v);
      double alphar = alpha.real();
      double alphai = alpha.imag();
      saxp<<<numunk/numproc/256,256>>>(&r_d[addres],&v[addres],&s[addres],make_cuDoubleComplex(-1*alphar,-1*alphai));
      //MATVEC 2 START
      matvec(s,o_d,t,ishermitian);
      nummatvec++;
      //MATVEC 2 FINISH
      //STABILIZER
      omega = inner(t,s)/norm2(t);
      complex<double> lan1 = inner(t,s);
      double lan2 = norm2(t);
      if(abs(lan1) < 1e-200 && lan2 < 1e-200)omega=1.0;//PREVENT NAN
      //UPDATE
      saxp<<<numunk/numproc/256,256>>>(&x_d[addres],&p[addres],&x_d[addres],make_cuDoubleComplex(alphar,alphai));
      double omegar = omega.real();
      double omegai = omega.imag();
      saxp<<<numunk/numproc/256,256>>>(&x_d[addres],&s[addres],&x_d[addres],make_cuDoubleComplex(omegar,omegai));
      saxp<<<numunk/numproc/256,256>>>(&s[addres],&t[addres],&r_d[addres],make_cuDoubleComplex(-1*omegar,-1*omegai));
      rnrm2 = sqrt(norm2(r_d));
      error = rnrm2/bnrm2;
      //if(myid_world==0)printf("RES. ERROR: %e ITER: %d\n",error,iter);
      if(error < res_tol)
        break;
      if(abs(omega) < 1e-200){
        printf("OMEGA BREAKDOWN\n");
        break;
      }
      rho_1 = rho;
    }
  }
  if(error < res_tol)
    ;//printf("CONVERGED!\n");
  else
    printf("NOT CONVERGED! **************************** iter: %d proc: %d\n",iter,myid);
  if(myid_world==0)printf("NUMBER OF %d ITERATIONS: %d (%d)\n",ishermitian,iter,nummatvec);
  //if(myid_world==0)printf("RESIDUAL ERROR NORM: %e\n",error);
  cudaMemcpy(&x[addres],&x_d[addres],numunk/numproc*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
  bicgst = bicgst + MPI_Wtime()-timet;
}
