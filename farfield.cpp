//Linear Solution Routines
//Mert Hidayetoglu, Oct 2015
#include "vars.h"

void farfield(complex<double> *x){

  extern int level;
  extern int box;

  extern int ninter;
  extern int *numsamp;
  extern int *numclus;

  extern complex<double> *coeff_multi;
  extern complex<double> **aggmulti;
  extern double **interp;
  extern int **intind;
  extern complex<double> **shiftmul;

  //LOWEST-LEVEL AGGREGATION
  #pragma omp parallel for
  for(int clusm = 0; clusm < numclus[level-1]; clusm++){
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
  for(int i = level-2; i > -1; i--){
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
    printf("level %d\n",i);
  }
  //INTERPOLATION
  int numsampf = 36000;
  double *interpf = new double[numsampf*ninter];
  int *intindf = new int[numsampf*ninter];
  complex<double> *pattern = new complex<double>[numsampf];
  double ratio = (double)numsamp[0]/numsampf;
  #pragma omp parallel for
  for(int m = 0; m < numsampf; m++){
    int center = 0;
    if(ninter%2 == 0)
      center = ceil(m*ratio);
    else
      center = round(m*ratio);
    double xm = (double)m/numsampf;
    for(int n = 0; n < ninter; n++){
      int ind = center+n-ninter/2;
      double mul = 1;
      double xn = (double)ind/numsamp[0];
      for(int k = 0; k < ninter; k++)
        if(k != n){
          double xk = (double)(center+k-ninter/2)/numsamp[0];
          mul = mul*(xm-xk)/(xn-xk);
        }
      interpf[m*ninter+n]=mul;
      ind = ind%numsamp[0];
      if(ind < 0)
        ind = ind + numsamp[0];
      intindf[m*ninter+n] = ind;
    }
  }
  for(int m = 0; m < numsampf; m++){
    complex<double> reduce = 0;
    for(int k = 0; k < ninter; k++)
      reduce = reduce + interpf[m*ninter+k]*aggmulti[0][intindf[m*ninter+k]];
    pattern[m] = reduce;
  }
  FILE *test = fopen("far.bin","wb");
  fwrite(pattern,sizeof(complex<double>),numsampf,test);
  fclose(test);
  delete[] interpf;
  delete[] intindf;
}
