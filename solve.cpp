//Linear Solution Routines
//Mert Hidayetoglu, Oct 2015
#include "vars.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void mlfmah(complex<double> *x, complex<double> *r){
  extern MPI_Comm MPI_COMM_MLFMA;
  extern int numunk;
  int myid;
  int numproc;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  #pragma omp parallel for
  for(int n = myid*numunk/numproc; n < (myid+1)*numunk/numproc; n++)
    x[n] = conj(x[n]);
  mlfma(x,r);
  #pragma omp parallel for
  for(int n = myid*numunk/numproc; n < (myid+1)*numunk/numproc; n++){
    r[n] = conj(r[n]);
    x[n] = conj(x[n]);
  }
}
void mlfma(complex<double> *x, complex<double> *r){

  extern int level;
  extern int box;

  extern MPI_Comm MPI_COMM_MLFMA;
  extern int ninter;
  extern int *numsamp;
  extern int *numclus;

  extern int **clusnear;
  extern int **clusfar;
  extern complex<double> *near;
  extern complex<double> *coeff_multi;
  extern complex<double> *basis_local;
  extern complex<double> **trans;
  extern int **traid;
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
  extern double innert;

  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  int numproc_g;
  int myid_g;
  MPI_Comm_size(MPI_COMM_WORLD,&numproc_g);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid_g);

  extern int **sendmap;
  extern int **recvmap;
  extern int **procmap;
  extern int **clusmap;
  extern int **procint;
  extern int *clcount;
  extern complex<double> **sendbuff;
  extern complex<double> **recvbuff;

  double timet = MPI_Wtime();
  matvecin++;

  //INITIALIZATION
  for(int i = 0; i < level; i++)
    #pragma omp parallel for
    for(int clusm = myid*numclus[i]/numproc; clusm < (myid+1)*numclus[i]/numproc; clusm++)
      fill_n(&agglocal[i][clusm*numsamp[i]],numsamp[i],complex<double>(0,0));
  //LOWEST-LEVEL AGGREGATION
  #pragma omp parallel for
  for(int clusm = myid*numclus[level-1]/numproc; clusm < (myid+1)*numclus[level-1]/numproc; clusm++){
    int indmulti = clusm*numsamp[level-1];
    int unk = clusm*box*box;
    for(int k = 0; k < numsamp[level-1]; k++){
      complex<double> reduce = 0;
      int indbasis = k*box*box;
      for(int n = 0; n < box*box; n++)
        reduce=reduce+coeff_multi[indbasis+n]*x[unk+n];
      aggmulti[level-1][indmulti+k]=reduce;
    }
  }
  //HIGHER-LEVEL AGGREGATIONS
  for(int i = level-2; i > 1; i--){
    #pragma omp parallel for
    for(int clusm = myid*numclus[i]/numproc; clusm < (myid+1)*numclus[i]/numproc; clusm++){
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
  for(int i = 2; i < level; i++)
    #pragma omp parallel for
    for(int cl = 0; cl < clcount[i]; cl++)
      memcpy(&sendbuff[i][cl*numsamp[i]],&aggmulti[i][sendmap[i][cl]*numsamp[i]],numsamp[i]*sizeof(complex<double>));
  //FARFIELD COMMUNICATION
  for(int i = 2; i < level; i++){
    for(int p = 0; p < numproc; p++){
      int sib = procmap[i][p];
      if(sib != -1){
        complex<double> *send = &sendbuff[i][procint[i][p]*numsamp[i]];
        complex<double> *recv = &recvbuff[i][procint[i][p]*numsamp[i]];
        int amount = clusmap[i][p]*numsamp[i];
        MPI_Sendrecv(send,amount,MPI_DOUBLE_COMPLEX,sib,0,recv,amount,MPI_DOUBLE_COMPLEX,sib,0,MPI_COMM_MLFMA,MPI_STATUS_IGNORE);
      }
    }
  }
  for(int i = 2; i < level; i++)
    #pragma omp parallel for
    for(int cl = 0; cl < clcount[i]; cl++)
      memcpy(&aggmulti[i][recvmap[i][cl]*numsamp[i]],&recvbuff[i][cl*numsamp[i]],numsamp[i]*sizeof(complex<double>));
  //TRANSLATION
  for(int i = 2; i < level; i++){
    #pragma omp parallel for
    for(int clusm = myid*numclus[i]/numproc; clusm < (myid+1)*numclus[i]/numproc; clusm++){
      int indlocal = clusm*numsamp[i];
      for(int cn = 0; cn < 27; cn++){
        int clusn = clusfar[i][clusm*27+cn];
        if(clusn!=-1){
          int indmulti = clusn*numsamp[i];
          int index = traid[i][clusm*27+cn]*numsamp[i];
          for(int k = 0; k < numsamp[i]; k++)
            agglocal[i][indlocal+k]=agglocal[i][indlocal+k]+trans[i][index+k]*aggmulti[i][indmulti+k];
        }
      }
    }
  }
  //HIGHER-LEVEL DISAGGREGATIONS
  for(int i = 2; i < level-1; i++){
    #pragma omp parallel for
    for(int clusn = myid*numclus[i]/numproc; clusn < (myid+1)*numclus[i]/numproc; clusn++){
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
          agglocal[i+1][indm+km]=agglocal[i+1][indm+km]+reduce;
        }
      }
    }
  }
  //LOWEST-LEVEL DISAGGREGATION
  #pragma omp parallel for
  for(int clusm = myid*numclus[level-1]/numproc; clusm < (myid+1)*numclus[level-1]/numproc; clusm++){
    int unk = clusm*box*box;
    int indlocal = clusm*numsamp[level-1];
    for(int n = 0; n < box*box; n++){
      complex<double> reduce = 0;
      int indbasis = n*numsamp[level-1];
      for(int k = 0; k < numsamp[level-1]; k++)
        reduce=reduce+basis_local[indbasis+k]*agglocal[level-1][indlocal+k];
      r[unk+n] = reduce;
    }
  }
  #pragma omp parallel for
  for(int cl = 0; cl < clcount[level]; cl++)
    memcpy(&sendbuff[level][cl*box*box],&x[sendmap[level][cl]*box*box],box*box*sizeof(complex<double>));
  //NEARFIELD COMMUNICATION
  for(int p = 0; p < numproc; p++){
    int sib = procmap[level][p];
    if(sib != -1){
      complex<double> *send = &sendbuff[level][procint[level][p]*box*box];
      complex<double> *recv = &recvbuff[level][procint[level][p]*box*box];
      int amount = clusmap[level][p]*box*box;
      MPI_Sendrecv(send,amount,MPI_DOUBLE_COMPLEX,sib,0,recv,amount,MPI_DOUBLE_COMPLEX,sib,0,MPI_COMM_MLFMA,MPI_STATUS_IGNORE);
    }
  }
  #pragma omp parallel for
  for(int cl = 0; cl < clcount[level]; cl++)
    memcpy(&x[recvmap[level][cl]*box*box],&recvbuff[level][cl*box*box],box*box*sizeof(complex<double>));
  //NEARFIELD
  #pragma omp parallel for
  for(int clusm = myid*numclus[level-1]/numproc; clusm < (myid+1)*numclus[level-1]/numproc; clusm++){
    int testing = clusm*box*box;
    for(int m = 0; m < box*box; m++){
      complex<double> reduce = 0;
      for(int cn = 0; cn < 9; cn++){
        int clusn = clusnear[level-1][clusm*9+cn];
        if(clusn != -1){
          int basis = clusn*box*box;
          int indbox = cn*box*box*box*box+m*box*box;
          for(int n = 0; n < box*box; n++)
            reduce=reduce+x[basis+n]*near[indbox+n];
        }
      }
      r[testing+m]=r[testing+m]+reduce;
    }
  }

  /*if(myid==numproc-1){
  int addres = myid*numclus[2]/numproc*numsamp[2];
  FILE *testf = fopen("agg_cpu.txt","w");
  for(int n = 0; n < numclus[2]/numproc*numsamp[2]; n++)
    fprintf(testf,"%e %e\n",aggmulti[2][addres+n].real(),aggmulti[2][addres+n].imag());
  fclose(testf);
  testf = fopen("loc_cpu.txt","w");
  for(int n = 0; n < numclus[2]/numproc*numsamp[2]; n++)
    fprintf(testf,"%e %e\n",agglocal[2][addres+n].real(),agglocal[2][addres+n].imag());
  fclose(testf);
  addres = myid*numclus[3]/numproc*numsamp[3];
  testf = fopen("loc_cpu_low.txt","w");
  for(int n = 0; n < numclus[3]/numproc*numsamp[3]; n++)
    fprintf(testf,"%e %e\n",agglocal[3][addres+n].real(),agglocal[3][addres+n].imag());
  fclose(testf);
  addres = myid*numclus[level-1]/numproc*numsamp[level-1];
  testf = fopen("loc_cpu_loww.txt","w");
  for(int n = 0; n < numclus[level-1]/numproc*numsamp[level-1]; n++)
    fprintf(testf,"%e %e\n",agglocal[level-1][addres+n].real(),agglocal[level-1][addres+n].imag());
  fclose(testf);
  extern int numunk;
  addres = myid*numunk/numproc;
  testf = fopen("unk.txt","w");
  printf("numunk %d numproc %d addres %d myid %d\n",numunk,numproc,addres,myid);
  for(int n = 0; n < numunk/numproc; n++)
    fprintf(testf,"%e %e\n",r[addres+n].real(),r[addres+n].imag());
  fclose(testf);
  }*/

  innert = innert + MPI_Wtime()-timet;
}
void aggregate(complex<double> *x, complex<double> *r){
  extern int level;
  extern int box;
  extern int ninter;
  extern int *numsamp;
  extern int *numclus;
  extern complex<double> *coeff_multi;
  extern complex<double> **aggmulti;
  extern complex<double> **agglocal;
  extern double **interp;
  extern int **intind;
  extern complex<double> **shiftmul;
  extern int matvecout;
  extern double outert;
  extern MPI_Comm MPI_COMM_MLFMA;
  int myid;
  int numproc;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
  double timet = MPI_Wtime();
  matvecout++;
  //LOWEST-LEVEL AGGREGATION
  #pragma omp parallel for
  for(int clusm = myid*numclus[level-1]/numproc; clusm < (myid+1)*numclus[level-1]/numproc; clusm++){
    int indmulti = clusm*numsamp[level-1];
    int unk = clusm*box*box;
    for(int k = 0; k < numsamp[level-1]; k++){
      complex<double> reduce = 0;
      int indbasis = k*box*box;
      for(int n = 0; n < box*box; n++)
        reduce=reduce+coeff_multi[indbasis+n]*x[unk+n];
      aggmulti[level-1][indmulti+k]=reduce;
    }
  }
  //HIGHER-LEVEL AGGREGATIONS
  for(int i = level-2; i > 1; i--){
    #pragma omp parallel for
    for(int clusm = myid*numclus[i]/numproc; clusm < (myid+1)*numclus[i]/numproc; clusm++){
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
  int amount = numclus[2]*numsamp[2]/numproc;
  int addres = myid*amount;
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
  /*extern complex<double> *go;
  extern int numrx;
  extern int numunk;
  int amount = numunk/numproc;
  int addres = myid*amount;
  #pragma omp parallel for
  for(int m = 0; m < numrx; m++){
    complex<double> reduce = 0;
    for(int n = 0; n < amount; n++)
      reduce = reduce + go[m*numunk+addres+n]*x[addres+n];
    r[m] = reduce;
  }
  complex<double> *rhs = new complex<double>[numrx];
  MPI_Allreduce(r,rhs,numrx,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_MLFMA);
  memcpy(r,rhs,numrx*sizeof(complex<double>));*/

  outert = outert + MPI_Wtime()-timet;
}
void aggregateh(complex<double> *x, complex<double> *r){
  extern int level;
  extern int box;
  extern int ninter;
  extern int *numsamp;
  extern int *numclus;
  extern complex<double> *basis_local;
  extern complex<double> **agglocal;
  extern double **anterp;
  extern int **antind;
  extern complex<double> **shiftloc;
  extern complex<double> **temp;
  extern int matvecout;
  extern double outert;
  extern double *anterph;
  extern int *antindh;
  extern double res;
  extern int numrx;
  extern MPI_Comm MPI_COMM_MLFMA;
  int myid;
  int numproc;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid);
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
  //HIGHER-LEVEL DISAGGREGATIONS
  for(int i = 2; i < level-1; i++){
    #pragma omp parallel for
    for(int clusn = myid*numclus[i]/numproc; clusn < (myid+1)*numclus[i]/numproc; clusn++){
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
  //LOWEST-LEVEL DISAGGREGATION
  #pragma omp parallel for
  for(int clusm = myid*numclus[level-1]/numproc; clusm < (myid+1)*numclus[level-1]/numproc; clusm++){
    int unk = clusm*box*box;
    int indlocal = clusm*numsamp[level-1];
    for(int n = 0; n < box*box; n++){
      complex<double> reduce = 0;
      int indbasis = n*numsamp[level-1];
      for(int k = 0; k < numsamp[level-1]; k++)
        reduce=reduce+basis_local[indbasis+k]*agglocal[level-1][indlocal+k];
      x[unk+n] = reduce;
    }
  }
  /*extern complex<double> *gh;
  extern int numrx;
  extern int numunk;
  int amount = numunk/numproc;
  int addres = myid*amount;
  #pragma omp parallel for
  for(int n = 0; n < amount; n++){
    complex<double> reduce = 0;
    for(int m = 0; m < numrx; m++)
      reduce = reduce + gh[(addres+n)*numrx+m]*r[m];
    x[addres+n] = reduce;
  }*/
  outert = outert + MPI_Wtime()-timet;
}
