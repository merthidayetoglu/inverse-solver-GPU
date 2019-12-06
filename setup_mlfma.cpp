#include "vars.h"

  int beta = 5;
  int ninter = 30;

  int *numclus;
  double *clusize;
  int *numsamp;
  int *numterm;

  complex<double> **clusc;
  int **clusnear;
  int **clusfar;
  int **cluspar;
  int **cluschi;
  int *unkpar;

  complex<double> *near;
  complex<double> *coeff_multi;
  complex<double> *basis_local;

  double **interp;
  double **anterp;
  int **intind;
  int **antind;
  complex<double> **shiftmul;
  complex<double> **shiftloc;
  complex<double> **trans;
  int **traid;
  complex<double> **aggmulti;
  complex<double> **agglocal;
  complex<double> **temp;
  //HIGHEST-LEVEL INTERPOLATORS
  double *interph;
  double *anterph;
  int *intindh;
  int *antindh;

  int **sendmap;
  int **recvmap;
  int **procmap;
  int **clusmap;
  int **procint;
  int *clcount;
  complex<double> **sendbuff;
  complex<double> **recvbuff;

  void setup_interp();

void setup_mlfma(){

  extern MPI_Comm MPI_COMM_MLFMA;
  extern double k0;
  extern int level;
  extern int box;
  extern double res;
  extern int numunk;
  extern complex<double> *pos;
  extern int *unkmap;

  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  int numproc_mlfma;
  int myid_mlfma;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc_mlfma);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid_mlfma);

  numclus = new int[level];
  clusize = new double[level];
  numsamp = new int[level];
  numterm = new int[level];

  for(int i = 0; i < level; i++)
    numclus[i] = pow(4,i);
  for(int i = 0; i < level; i++)
    clusize[i] = res*box*pow(2,level-i-1);

  for(int i = 0; i < level; i++){
    double kr = sqrt(2)*k0*clusize[i];
    double term = kr+1.8*pow(beta,2.0/3)*pow(kr,1.0/3);
    numterm[i] = floor(term);
    numsamp[i] = 4*numterm[i];
  }

  if(myid==0){
    printf("NUM CLUS        :\n");
    for(int i = 0; i < level; i++)
      printf("LEVEL %d: %d\n",i+1,numclus[i]);
    for(int i = 0; i < level; i++)
      printf("LEVEL: %d  CLUSSIZE: %f  NUM. SAMP: %d  TOTAL: %d\n",i+1,clusize[i],numsamp[i],numsamp[i]*numclus[i]);
    printf("NUM TERMS       : \n");
    for(int i = 0; i < level; i++)
      printf("LEVEL: %d  NUM.TERM: %d\n",i+1,numterm[i]);
  }
  if(myid==0)
    #pragma omp parallel
    if(omp_get_thread_num()==0){
      printf("NUM. PROCS: %d, NUM. THREADS: %d\n",numproc,omp_get_num_threads());
      printf("NUM. MLFMA PROCS: %d\n",numproc_mlfma);
    }

  //PREPROCESSING
  pos = new complex<double>[numunk];
  clusc = new complex<double>*[level];
  clusnear = new int*[level];
  clusfar = new int*[level];
  cluspar = new int*[level];
  cluschi = new int*[level];
  for(int i = 0; i < level; i++){
    clusc[i] = new complex<double>[numclus[i]];
    clusnear[i] = new int[numclus[i]*9];
    clusfar[i] = new int[numclus[i]*27];
    cluspar[i] = new int[numclus[i]];
    if(i < level-1)
      cluschi[i] = new int[numclus[i]*4];
    else
      cluschi[i] = new int[numclus[i]*box*box];
  }
  unkpar = new int[numunk];
  unkmap = new int[numunk];

  //TREE STRUCTURE
  cluspar[0][0]=-1;
  for(int i = 0; i < level; i++){
    int lenpar = pow(2,i);
    #pragma omp parallel for
    for(int m = 0; m < lenpar; m++)
      for(int n = 0; n < lenpar; n++){
        int par = m*lenpar+n;
        //HIGHER LEVELS
        if(i < level-1){
          for(int k = 0; k < 2; k++)
            for(int l = 0; l < 2; l++){
              //int chi = m*lenchi*2+n*2+lenchi*k+l;
              int chi = par*4+k*2+l;
              cluschi[i][par*4+k*2+l]=chi;
              cluspar[i+1][chi]=par;
            }
        }
        //LOWEST LEVEL
        else{
          for(int k = 0; k < box; k++)
            for(int l = 0; l < box; l++){
              //int chi = m*lenchi*box+n*box+lenchi*k+l;
              int chi = par*box*box+k*box+l;
              cluschi[i][par*box*box+k*box+l]=chi;
              unkpar[chi]=par;
            }
        }
      }
  }
  if(myid==0)printf("COORDINATES\n");
  //CLUSTER COORDINATES
  clusc[0][0]=0;
  for(int i = 0; i < level; i++){
    #pragma omp parallel for
    for(int m = 0; m < numclus[i]; m++){
      if(i < level-1){
        complex<double>cor=clusc[i][m]+clusize[i]/4*complex<double>(-1,+1);
        for(int k = 0; k < 2; k++)
          for(int l = 0; l < 2; l++)
            clusc[i+1][cluschi[i][m*4+k*2+l]]=cor+complex<double>(l*clusize[i+1],-k*clusize[i+1]);
      }
      //LOWEST LEVEL
      else{
        complex<double>cor=clusc[i][m]+(clusize[i]/2-res/2)*complex<double>(-1,+1);
        for(int k = 0; k < box; k++)
          for(int l = 0; l < box; l++){
            pos[cluschi[i][m*box*box+k*box+l]]=cor+complex<double>(l*res,-k*res);
          }
      }
    }
  }
  if(myid==0)printf("MAPPING\n");
  //UNKNOWN MAPPING
  complex<double> cor = clusize[0]/2*complex<double>(-1,1);
  int len = box*pow(2,level-1);
  #pragma omp parallel for
  for(int n = 0; n < numunk; n++){
    int l = floor((pos[n].real()-cor.real())/res);
    int k = floor((cor.imag()-pos[n].imag())/res);
    unkmap[k*len+l]=n;
  }
  if(myid==0)printf("NEAR-FAR\n");
  //NEAR-FIELD & FAR-FIELD CLUSTERS
  for(int i = 0; i < level; i++){
    #pragma omp parallel for
    for(int m = 0; m < numclus[i]; m++){
      for(int n = 0; n < 9; n++)
        clusnear[i][m*9+n]=-1;
      for(int n = 0; n < 27; n++)
        clusfar[i][m*27+n]=-1;
    }
  }
  clusnear[0][0]=0;
  for(int i = 1; i < level; i++){
    double clusdim = clusize[i];
    #pragma omp parallel for
    for(int m = 0; m < numclus[i]; m++){
      int nearid = 0;
      int farid = 0;
      complex<double>posm=clusc[i][m];
      int par = cluspar[i][m];
      for(int ind = 0; ind < 9; ind++){
        int uncle = clusnear[i-1][par*9+ind];
        if(uncle != -1){
          for(int ind2 = 0; ind2 < 4; ind2++){
            int n=cluschi[i-1][uncle*4+ind2];
            if(abs(clusc[i][n]-posm) < clusdim*sqrt(3)){
              clusnear[i][m*9+nearid]=n;
              nearid++;
            }
            else{
              clusfar[i][m*27+farid]=n;
              farid++;
            }
          }
        }
      }
    }
  }
  //MPI SETUP
  if(myid==0)printf("MPI SETUP\n");
  int *proctemp = new int[numproc_mlfma];
  int **friends = new int*[level+1];
  int **clustemp = new int*[level+1];
  for(int i = 0; i < level; i++){
    friends[i] = new int[numproc_mlfma];
    clustemp[i] = new int[numclus[i]];
    fill_n(friends[i],numproc_mlfma,0);
    fill_n(clustemp[i],numclus[i],-1);
  }
  friends[level] = new int[numproc_mlfma];
  clustemp[level] = new int[numclus[level-1]];
  fill_n(friends[level],numproc_mlfma,0);
  fill_n(clustemp[level],numclus[level-1],-1);
  //CLUSTEMP & FRIENDS
  for(int i = 0; i < level; i++){
    for(int clusm = 0; clusm < numclus[i]/numproc_mlfma; clusm++)
      for(int cn = 0; cn < 27; cn++){
        int clusn = clusfar[i][(numclus[i]/numproc_mlfma*myid_mlfma+clusm)*27+cn];
        if(clusn!=-1)clustemp[i][clusn] = clusn/(numclus[i]/numproc_mlfma);
      }
    for(int m = 0; m < numclus[i]; m++){
      int p = clustemp[i][m];
      if(p!=-1)friends[i][p]++;
    }
  }
  for(int clusm = 0; clusm < numclus[level-1]/numproc_mlfma; clusm++)
    for(int cn = 0; cn < 9; cn++){
      int clusn = clusnear[level-1][(numclus[level-1]/numproc_mlfma*myid_mlfma+clusm)*9+cn];
      if(clusn!=-1)clustemp[level][clusn] = clusn/(numclus[level-1]/numproc_mlfma);
    }
  for(int m = 0; m < numclus[level-1]; m++){
    int p = clustemp[level][m];
    if(p!=-1)friends[level][p]++;
  }
  sendmap = new int*[level+1];
  recvmap = new int*[level+1];
  procint = new int*[level+1];
  procmap = new int*[level+1];
  sendbuff = new complex<double>*[level+1];
  recvbuff = new complex<double>*[level+1];
  clusmap = new int*[level+1];
  clcount = new int[level+1];
  for(int i = 0; i < level+1; i++){
    procint[i] = new int[numproc_mlfma];
    procmap[i] = new int[numproc_mlfma];
    clusmap[i] = new int[numproc_mlfma];
    fill_n(procint[i],numproc_mlfma,-1);
    fill_n(procmap[i],numproc_mlfma,-1);
    fill_n(clusmap[i],numproc_mlfma,0);
  }
  //PROCTEMP
  {
    int count;
    int me,sib;
    char strtemp[80];
    if(numproc_mlfma>9)
      sprintf(strtemp,"internodes0%d.dat",numproc_mlfma);
    else
      sprintf(strtemp,"internodes00%d.dat",numproc_mlfma);
    if(myid==0)printf("FILE %s\n",strtemp);
    FILE *mapf = fopen(strtemp,"r");
    fscanf(mapf,"%d",&count);
    count = 0;
    proctemp[count]=myid_mlfma;
    for(int i = 1; i < numproc_mlfma; i++)
      for(int n = 0; n < numproc_mlfma; n++){
        fscanf(mapf,"%d %d",&me,&sib);
        if(me == myid_mlfma){
          count++;
          proctemp[count] = sib;
        }
      }
    fclose(mapf);
  }
  //PROCMAP
  if(myid==0)printf("PROCMAP\n");
  for(int i = 0; i < level+1; i++){
    int numsib = 0;
    int numcls = 0;
    for(int p = 0; p < numproc_mlfma; p++){
      if(friends[i][proctemp[p]] > 0 && proctemp[p]!=myid_mlfma){
        procmap[i][numsib] = proctemp[p];
        clusmap[i][numsib] = friends[i][proctemp[p]];
        numcls = numcls + friends[i][proctemp[p]];
        numsib++;
      }
    }
    clcount[i] = numcls;
  }
  if(myid==0){
    printf("PROCMAP %d\n",myid_mlfma);
    for(int i = 0; i < level+1; i++){
      for(int p = 0; p < numproc_mlfma; p++)
        printf("%d ",procmap[i][p]);
      printf("\n");
    }
    printf("CLUSMAP %d\n",myid_mlfma);
    for(int i = 0; i < level; i++){
      for(int p = 0; p < numproc_mlfma; p++)
        printf("%d ",clusmap[i][p]);
      printf("| %d | %d\n",clcount[i],clcount[i]*numsamp[i]);
    }
    for(int p = 0; p < numproc_mlfma; p++)
      printf("%d ",clusmap[level][p]);
    printf("| %d | %d\n",clcount[level],clcount[level]*box*box);
  }
  for(int i = 2; i < level+1; i++){
    recvmap[i] = new int[clcount[i]];
    sendmap[i] = new int[clcount[i]];
    if(i < level){
      sendbuff[i] = new complex<double>[clcount[i]*numsamp[i]];
      recvbuff[i] = new complex<double>[clcount[i]*numsamp[i]];
    }
    else{
      sendbuff[i] = new complex<double>[clcount[i]*box*box];
      recvbuff[i] = new complex<double>[clcount[i]*box*box];
    }
  }
  //RECVMAP
  if(myid==0)printf("RECVMAP\n");
  for(int i = 0; i < level+1; i++){
    int count = 0;
    for(int p = 0; p < numproc_mlfma; p++){
      int procm = procmap[i][p];
      if(procm != -1){
        procint[i][p] = count;
        if(i < level)
          for(int m = 0; m < numclus[i]/numproc_mlfma; m++){
            int clusn = clustemp[i][numclus[i]/numproc_mlfma*procm+m];
            if(clusn != -1){
              recvmap[i][count] = procm*numclus[i]/numproc_mlfma+m;
              count++;
            }
          }
        else
          for(int m = 0; m < numclus[level-1]/numproc_mlfma; m++){
            int clusn = clustemp[i][numclus[level-1]/numproc_mlfma*procm+m];
            if(clusn != -1){
              recvmap[i][count] = procm*numclus[level-1]/numproc_mlfma+m;
              count++;
            }
          }
      }
    }
  }
  if(myid==0){
    printf("PROCINT %d\n",myid_mlfma);
    for(int i = 0; i < level+1; i++){
      for(int p = 0; p < numproc_mlfma; p++)
        printf("%d ",procint[i][p]);
      printf("\n");
    }
    printf("RECVMAP %d\n",myid_mlfma);
    for(int i = 0; i < 5; i++){
      for(int m = 0; m < clcount[i]; m++)
        printf("%d ",recvmap[i][m]);
      printf(" **********\n");
    }
  }
  //SENDMAP
  for(int i = 2; i < level+1; i++){
    if(myid==0)printf("SENDMAP  %d\n",i+1);
    for(int p = 0; p < numproc_mlfma; p++){
      int sib = procmap[i][p];
      if(sib != -1){
        int *send = &recvmap[i][procint[i][p]];
        int *recv = &sendmap[i][procint[i][p]];
        int amount = clusmap[i][p];
        MPI_Sendrecv(send,amount,MPI_INTEGER,sib,0,recv,amount,MPI_INTEGER,sib,0,MPI_COMM_MLFMA,MPI_STATUS_IGNORE);
      }
    }
  }
  if(myid==0){
    printf("SENDMAP %d\n",myid_mlfma);
    for(int i = 0; i < 5; i++){
      for(int m = 0; m < clcount[i]; m++)
        printf("%d ",sendmap[i][m]);
      printf(" **********\n");
    }
  }
  for(int i = 0; i < level; i++){
    delete[] friends[i];
    delete[] clustemp[i];
  }
  delete[] proctemp;
  delete[] friends;
  delete[] clustemp;
  //SETUP
  complex<double> *hank = new complex<double>[2*numterm[0]+1];
  complex<double> *expn = new complex<double>[numterm[0]];
  int *list = new int[numsamp[0]];
  int *neid = new int[numclus[level-1]*9];
  near = new complex<double>[9*box*box*box*box];
  coeff_multi = new complex<double>[numsamp[level-1]*box*box];
  basis_local = new complex<double>[box*box*numsamp[level-1]];
  interp = new double*[level];
  anterp = new double*[level];
  intind = new int*[level];
  antind = new int*[level];
  shiftmul = new complex<double>*[level];
  shiftloc = new complex<double>*[level];
  trans = new complex<double>*[level];
  traid = new int*[level];
  for(int i = 0; i < level; i++){
    interp[i] = new double[numsamp[i]*ninter];
    anterp[i] = new double[numsamp[i]*2*ninter];
    intind[i] = new int[numsamp[i]*ninter];
    antind[i] = new int[numsamp[i]*2*ninter];
    shiftmul[i] = new complex<double>[4*numsamp[i]];
    shiftloc[i] = new complex<double>[4*numsamp[i]];
    trans[i] = new complex<double>[49*numsamp[i]];
    traid[i] = new int[numclus[i]*27];
  }
  aggmulti = new complex<double>*[level];
  agglocal = new complex<double>*[level];
  for(int i = 0; i < level; i++){
    aggmulti[i] = new complex<double>[numclus[i]*numsamp[i]];
    agglocal[i] = new complex<double>[numclus[i]*numsamp[i]];
  }
  #pragma omp parallel shared(temp)
  {
    if(omp_get_thread_num() == 0){
      temp = new complex<double>*[omp_get_num_threads()];
      for(int i = 0; i < omp_get_num_threads(); i++)
        temp[i] = new complex<double>[numsamp[0]];
    }
  }
  //FILL NEARFIELD MATRIX
  if(myid==0)printf("NEARFIELD MATRIX\n");
  #pragma omp parallel for
  for(int n = 0; n < box*box*box*box; n++)
    for(int m = 0; m < 9; m++)
      near[n*9+m]=0;
  #pragma omp parallel for
  for(int clusm = 0; clusm < numclus[level-1]; clusm++)
    for(int cn = 0; cn < 9; cn++){
      int clusn = clusnear[level-1][clusm*9+cn];
      if(clusn != -1){
        complex<double> m2l = clusc[level-1][clusm]-clusc[level-1][clusn];
        int m = round(m2l.imag()/clusize[level-1])+1;
        int n = round(-m2l.real()/clusize[level-1])+1;
        int ind = m*3+n;
        neid[clusm*9+cn]=ind;
      }
      else
        neid[clusm*9+cn]=-1;
    }
  //RESORT
  #pragma omp parallel for
  for(int clusm = 0; clusm < numclus[level-1]; clusm++){
    int clist[9] = {-1,-1,-1,-1,-1,-1,-1,-1,-1};
    for(int cn = 0; cn < 9; cn++){
      int id = neid[clusm*9+cn];
      if(id != -1)
        clist[id] = clusnear[level-1][clusm*9+cn];
    }
    for(int cn = 0; cn < 9; cn++)clusnear[level-1][clusm*9+cn]=clist[cn];
    //memcpy(&clusnear[level-1][clusm*9],&clist[0],9*sizeof(int));
  }
  if(myid==0){
    for(int m = 0; m < 16; m++){
      for(int n = 0; n < 9; n++)
        printf("%d ",clusnear[level-1][m*9+n]);
      printf("\n");
    }
  }
  #pragma omp parallel for
  for(int clusm = 0; clusm < numclus[level-1]; clusm++)
    for(int cn = 0; cn < 9; cn++){
      int clusn = clusnear[level-1][clusm*9+cn];
      if(clusn != -1)
        if(near[cn*box*box*box*box]==complex<double>(0))
          for(int m = 0; m < box*box*box*box; m++){
            int k = m/(box*box);
            int l = m%(box*box);
            int testing = cluschi[level-1][clusm*box*box+k];
            int basis = cluschi[level-1][clusn*box*box+l];
            near[cn*box*box*box*box+k*box*box+l]=integrate(pos[testing],pos[basis]);
          }
    }
  if(myid==0)printf("TRANSLATION OPERATORS\n");
  #pragma omp parallel for
  for(int p = 0; p < numterm[0]; p++)
    expn[p] = exp(complex<double>(0,(p+1)*M_PI/2));
  //FILL TRANSLATION OPERATORS
  for(int i = 2; i < level; i++){
    if(myid==0)printf("level %d\n",i+1);
    double a = clusize[i];
    #pragma omp parallel for
    for(int clusm = 0; clusm < numclus[i]; clusm++)
      for(int cn = 0; cn < 27; cn++){
        int clusn = clusfar[i][clusm*27+cn];
        if(clusn != -1){
          complex<double> l2m = clusc[i][clusn]-clusc[i][clusm];
          int n = round(l2m.real()/a)+3;
          int m = round(-l2m.imag()/a)+3;
          int ind = m*7+n;
          traid[i][clusm*27+cn]=ind;
        }
      }
    for(int cm = 0; cm < 7; cm++)
      for(int cn = 0; cn < 7; cn++){
        complex<double> l2m = -complex<double>(-3*a+cn*a,3*a-cm*a);
        if(abs(l2m)>a*sqrt(3)){
          #pragma omp parallel for
          for(int p = 0; p < numterm[i]+1; p++)
            hank[p] = hn(p,k0*abs(l2m));
          int index = cm*7*numsamp[i]+cn*numsamp[i];
          #pragma omp parallel for
          for(int k = 0; k < numsamp[i]; k++){
            complex<double> reduce = hank[0];
            for(int p = 1; p < numterm[i]+1; p++)
              reduce=reduce+2*cos(p*(2*M_PI*k/numsamp[i]-arg(l2m)))*hank[p]*expn[p-1];
            trans[i][index+k] = reduce;
          }
        }
      }
  }
  //FILL COEFF FOR MULTIPOLE
  //FILL BASIS FOR LOCAL
  #pragma omp parallel for
  for(int n = 0; n < box*box; n++){
    int unk = cluschi[level-1][n];
    for(int k = 0; k < numsamp[level-1]; k++){
      coeff_multi[k*box*box+n] = integrate_multi(clusc[level-1][0],pos[unk],k);
      basis_local[n*numsamp[level-1]+k]=integrate_local(clusc[level-1][0],pos[unk],k)/complex<double>(numsamp[level-1])*complex<double>(0,0.25);
    }
  }
  //FILL SHIFTERS
  if(myid==0)printf("MULTIPOLE & LOCAL SHIFTERS\n");
  for(int i = 0; i < level-1; i++){
    int par = 0;
    for(int cn = 0; cn < 4; cn++){
      int chi = cluschi[i][par*numclus[i]+cn];
      complex<double> rho = clusc[i][par]-clusc[i+1][chi];
      #pragma omp parallel for
      for(int k = 0; k < numsamp[i]; k++){
        double ang = 2*M_PI*k/numsamp[i];
        complex<double> kw = k0*complex<double>(cos(ang),sin(ang));
        double product = kw.real()*rho.real()+kw.imag()*rho.imag();
        shiftmul[i][cn*numsamp[i]+k]=exp(complex<double>(0,product));
        shiftloc[i][cn*numsamp[i]+k]=exp(complex<double>(0,-product));
      }
    }
  }
  if(myid==0)printf("INTERPOLATORS\n");
  //FILL INTERPOLATORS
  for(int i = 0; i < level-1; i++){
    double ratio = (double)numsamp[i+1]/numsamp[i];
    #pragma omp parallel for
    for(int m = 0; m < numsamp[i]; m++){
      int center = 0;
      if(ninter%2 == 0)
        center = ceil(m*ratio);
      else
        center = round(m*ratio);
      double xm = (double)m/numsamp[i];
      for(int n = 0; n < ninter; n++){
        int ind = center+n-ninter/2;
        double mul = 1;
        double xn = (double)ind/numsamp[i+1];
        for(int k = 0; k < ninter; k++)
          if(k != n){
            double xk = (double)(center+k-ninter/2)/numsamp[i+1];
            mul = mul*(xm-xk)/(xn-xk);
          }
        interp[i][m*ninter+n]=mul;
        ind = ind%numsamp[i+1];
        if(ind < 0)
          ind = ind + numsamp[i+1];
        intind[i][m*ninter+n] = ind;
      }
    }
  }
  if(myid==0)printf("ANTERPOLATORS\n");
  int *tempi = new int[2*ninter];
  double *tempd = new double[2*ninter];
  //FILL ANTERPOLATORS
  for(int i = 1; i < level; i++){
    for(int n = 0; n < numsamp[0]; n++)
      list[n] = 0;
    for(int m = 0; m < numsamp[i]; m++)
      for(int n = 0; n < 2*ninter; n++){
        antind[i][m*2*ninter+n] = 0;
        anterp[i][m*2*ninter+n] = 0;
      }
    for(int m = 0; m < numsamp[i-1]; m++)
      for(int n = 0; n < ninter; n++){
        int ind = intind[i-1][m*ninter+n];
        antind[i][ind*2*ninter+list[ind]] = m;
        anterp[i][ind*2*ninter+list[ind]] = interp[i-1][m*ninter+n]*numsamp[i]/numsamp[i-1];
        list[ind]++;
      }
    for(int m = 0; m < numsamp[i]; m++)
      for(int n = 1; n < list[m]; n++)
        if(antind[i][m*2*ninter+n]!=antind[i][m*2*ninter+n-1]+1){
            //if(myid==0)printf("level %d sample %d index %d(%d) index %d(%d)\n",i,m,n,antind[i][m*2*ninter+n],n-1,antind[i][m*2*ninter+n-1]);
            memcpy(&tempi[0],&antind[i][m*2*ninter+n],(list[m]-n)*sizeof(int));
            memcpy(&tempd[0],&anterp[i][m*2*ninter+n],(list[m]-n)*sizeof(double));
            memcpy(&tempi[list[m]-n],&antind[i][m*2*ninter],n*sizeof(int));
            memcpy(&tempd[list[m]-n],&anterp[i][m*2*ninter],n*sizeof(double));
            memcpy(&antind[i][m*2*ninter],tempi,list[m]*sizeof(int));
            memcpy(&anterp[i][m*2*ninter],tempd,list[m]*sizeof(double));
            break;
          }
  }
  delete[] tempi;
  delete[] tempd;
  delete[] hank;
  delete[] list;
  delete[] expn;
  delete[] neid;

  setup_interp();
}

void setup_interp(){

  extern int numrx;
  extern int ninter;
  extern int *numsamp;
  interph = new double[numrx*ninter];
  anterph = new double[numsamp[0]*2*ninter];
  intindh = new int[numrx*ninter];
  antindh = new int[numsamp[0]*2*ninter];
  int *list = new int[numsamp[0]];
  //TOP-LEVEL INTERPOLATORS
  double ratio = (double)numsamp[0]/numrx;
  #pragma omp parallel for
  for(int m = 0; m < numrx; m++){
    int center = 0;
    if(ninter%2 == 0)
      center = ceil(m*ratio);
    else
      center = round(m*ratio);
    double xm = (double)m/numrx;
    for(int n = 0; n < ninter; n++){
      int ind = center+n-ninter/2;
      double mul = 1;
      double xn = (double)ind/numsamp[0];
      for(int k = 0; k < ninter; k++)
        if(k != n){
          double xk = (double)(center+k-ninter/2)/numsamp[0];
          mul = mul*(xm-xk)/(xn-xk);
        }
      interph[m*ninter+n]=mul;
      ind = ind%numsamp[0];
      if(ind < 0)
        ind = ind + numsamp[0];
      intindh[m*ninter+n] = ind;
    }
  }
  //TOP-LEVEL ANTERPOLATORS
  #pragma omp parallel for
  for(int m = 0; m < numsamp[0]; m++){
    list[m] = 0;
    for(int n = 0; n < 2*ninter; n++){
      antindh[m*2*ninter+n] = 0;
      anterph[m*2*ninter+n] = 0;
    }
  }
  for(int m = 0; m < numrx; m++)
    for(int n = 0; n < ninter; n++){
      int ind = intindh[m*ninter+n];
      antindh[ind*2*ninter+list[ind]] = m;
      anterph[ind*2*ninter+list[ind]] = interph[m*ninter+n]*numsamp[0]/numrx;
      list[ind]++;
    }
  delete[] list;
}
