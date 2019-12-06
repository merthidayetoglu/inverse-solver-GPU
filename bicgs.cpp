#include "vars.h"

extern MPI_Comm MPI_COMM_MLFMA;
extern int numunk;
extern int iti;
extern double toli;

complex<double> *p;
complex<double> *v;
complex<double> *s;
complex<double> *t;
complex<double> *r;
complex<double> *r_tld;
complex<double> *buffer;

extern double bicgst;

double norm2(complex<double> *a){
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  double ret = 0;
  for(int m = myid*numunk/numproc; m < (myid+1)*numunk/numproc; m++)
    ret = ret + norm(a[m]);
  double rettot;
  MPI_Allreduce(&ret,&rettot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_MLFMA);
  return rettot;
}
complex<double> inner(complex<double> *a, complex<double> *b){
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  complex<double> red = 0;
  for(int m = myid*numunk/numproc; m < (myid+1)*numunk/numproc; m++)
    red = red + conj(a[m])*b[m];
  complex<double> redtot;
  MPI_Allreduce(&red,&redtot,1,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_MLFMA);
  return redtot;
}
void saxpy(complex<double> *a, complex<double> *b, complex<double> *c, complex<double> alpha){//c = a + alpha*b
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  #pragma omp parallel for
  for(int n = myid*numunk/numproc; n < (myid+1)*numunk/numproc; n++)
    c[n] = a[n] + alpha*b[n];
}

void setup_bicgs(){
  //ALLOCATIONS
  p = new complex<double>[numunk];
  v = new complex<double>[numunk];
  s = new complex<double>[numunk];
  t = new complex<double>[numunk];
  r = new complex<double>[numunk];
  r_tld = new complex<double>[numunk];
  buffer = new complex<double>[numunk];
}
void matvec(complex<double> *x, complex<double> *o, complex<double> *b, bool ishermitian){
  extern int numunk;
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  int amount = numunk/numproc;
  int addres = myid*amount;
  if(ishermitian){
    #pragma omp parallel for
    for(int n = addres; n < addres+amount; n++)
      buffer[n] = conj(x[n]);
    mlfma(buffer,b);
    #pragma omp parallel for
    for(int n = addres; n < addres+amount; n++)
      b[n] = x[n]-conj(o[n]*b[n]);
  }else{
    #pragma omp parallel for
    for(int n = addres; n < addres+amount; n++)
      buffer[n] = o[n]*x[n];
    mlfma(buffer,b);
    #pragma omp parallel for
    for(int n = addres; n < addres+amount; n++)
      b[n] = x[n]-b[n];
  }
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
  bnrm2 = sqrt(norm2(b));
  if(bnrm2 < pow(10,-20))
    bnrm2 = 1;
  //MATVEC 0 START
  matvec(x,o,r,ishermitian);
  nummatvec++;
  //MATVEC 0 FINISH
  saxpy(b,r,r,complex<double>(-1,0));
  rnrm2 = sqrt(norm2(r));
  error = rnrm2/bnrm2;
  //if(myid_world==0)printf("RES. ERROR: %e ITER: %d\n",error,iter);
  if(error > res_tol){
    memcpy(&r_tld[myid*numunk/numproc],&r[myid*numunk/numproc],numunk/numproc*sizeof(complex<double>));
    //BEGIN ITERATIONS
    while(iter < max_it){
      iter++;
      rho = inner(r_tld,r);
      if(abs(rho) < 1e-20){
        printf("RHO BREAKDOWN\n");
        break;
      }
      if(iter == 1)
        memcpy(&p[myid*numunk/numproc],&r[myid*numunk/numproc],numunk/numproc*sizeof(complex<double>));
      else{
        beta = (rho/rho_1)*(alpha/omega);
        #pragma omp parallel for
        for(int n = myid*numunk/numproc; n < (myid+1)*numunk/numproc; n++)
          p[n] = r[n] + beta*(p[n]-omega*v[n]);
      }
      //PRECONDITIONER
      //MATVEC 1 START
      matvec(p,o,v,ishermitian);
      nummatvec++;
      //MATVEC 1 FINISH
      alpha = rho/inner(r_tld,v);
      saxpy(r,v,s,alpha*complex<double>(-1,0));

      //MATVEC 2 START
      matvec(s,o,t,ishermitian);
      nummatvec++;
      //MATVEC 2 FINISH
      //STABILIZER
      omega = inner(t,s)/norm2(t);
      //UPDATE
      #pragma omp parallel for
      for(int n = myid*numunk/numproc; n < (myid+1)*numunk/numproc; n++){
        x[n] = x[n] + alpha*p[n] + omega*s[n];
        r[n] = s[n] - omega*t[n];
      }
      rnrm2 = sqrt(norm2(r));
      error = rnrm2/bnrm2;
      //if(myid_world==0)printf("RES. ERROR: %e ITER: %d\n",error,iter);
      if(error < res_tol)
        break;
      if(abs(omega) < 1e-20){
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
  bicgst = bicgst + MPI_Wtime()-timet;
}
