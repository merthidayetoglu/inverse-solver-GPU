//Mert Hidayetoglu, Feb 2016
#include "vars.h"

  double res = 0.1;
  int box = 8;
  int level = 7;
  double k0; 

  int numfreq;
  int numrx;
  int numtx;
  int txproc;
  int myfreq;
  int mytx;
  MPI_Comm MPI_COMM_MLFMA;
  MPI_Comm MPI_COMM_DBIM;

  double dim;
  int numunk;

  double toli=1e-4;
  int iti=300;
  double tolo=1e-4;
  int ito=300;

  complex<double> *rx;
  complex<double> *tx;
  complex<double> *go;
  complex<double> *gh;
  double *scales;
  double scale;
  double regular = 0.0;

  complex<double> *o;
  complex<double> *od;
  complex<double> *inc;
  complex<double> *tot;
  complex<double> *pos;
  int *unkmap;

  double mesnorm;
  double objnorm;
  complex<double> *omes;
  complex<double> *mes;
  complex<double> *mesd;
  complex<double> *rhs;
  complex<double> *x;
  complex<double> *y;
  complex<double> *c;
  complex<double> *g;
  complex<double> *ot;
  complex<double> *buff;
  complex<double> *save;

  FILE *errhist;
  FILE *ithist;
  FILE *file;
  char *chartemp;
  int itfor;
  int itinv;
  int matvecin = 0;
  int matvecout = 0;

  double setupt;
  double solutt;
  double innert;
  double outert;
  double totalt;
  double bicgst;

int main(int argc, char** argv){

  int numproc;
  int myid;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  int numproc_mlfma;
  int myid_mlfma;

  MPI_Barrier(MPI_COMM_WORLD);
  totalt = MPI_Wtime();

  chartemp = getenv("NUMFREQ");
  numfreq = atoi(chartemp);
  chartemp = getenv("NUMTX");
  numtx = atoi(chartemp);
  chartemp = getenv("TXPROC");
  txproc = atoi(chartemp);
  chartemp = getenv("NUMRX");
  numrx = atoi(chartemp);
  chartemp = getenv("MLFMA");
  numproc_mlfma = atoi(chartemp);
  chartemp = getenv("MINFREQ");
  double minfreq = atof(chartemp);
  chartemp = getenv("MAXFREQ");
  double maxfreq = atof(chartemp);

  int myid_dbim = myid/numproc_mlfma;
  int numproc_dbim = numproc/numproc_mlfma;
  MPI_Comm_split(MPI_COMM_WORLD,myid_dbim,myid,&MPI_COMM_MLFMA);
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc_mlfma);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid_mlfma);
  MPI_Comm_split(MPI_COMM_WORLD,myid_mlfma,myid,&MPI_COMM_DBIM);
  MPI_Comm_size(MPI_COMM_DBIM,&numproc_dbim);
  MPI_Comm_rank(MPI_COMM_DBIM,&myid_dbim);
  printf("myid: %d/%d myid_dbim: %d/%d myid_mlfma: %d/%d\n",myid,numproc,myid_dbim,numproc_dbim,myid_mlfma,numproc_mlfma);


  mytx = (myid_dbim*txproc)%numtx;
  myfreq = myid_dbim/(numtx/txproc);
  //printf("myid: %d myid_dbim: %d myid_mlfma: %d mytx: %d myfreq: %d\n",myid,myid_dbim,myid_mlfma,mytx,myfreq);
  if(numproc != numtx/txproc*numfreq*numproc_mlfma){
    if(myid==0)printf("INCONSISTENT # PROCESSES\n");
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
  }

  if(myid==0)
    #pragma omp parallel
    if(omp_get_thread_num()==0){
      printf("NUM. PROCS: %d, NUM. THREADS: %d\n",numproc,omp_get_num_threads());
      printf("NUM. TX: %d NUM. FREQ: %d\n",numtx,numfreq);
      printf("NUM. TX/PROC: %d\n",txproc);
      printf("NUM. DBIM PROCS: %d\n",numproc_dbim);
      printf("NUM. MLFMA PROCS: %d\n",numproc_mlfma);
    }

  //FREQUENCIES
  scales = new double[numfreq];
  for(int i = 0; i < numfreq; i++){
    scale = maxfreq-(maxfreq-minfreq)/(numfreq-1)*i;
    if(numfreq==1)scale=maxfreq;
    scales[i] = scale;
  }
  k0 = 2*M_PI*scales[myfreq];
  dim = res*box*pow(2,level-1);
  numunk = pow(dim/res,2);

  //TRANSMITTERS (FARFIELD)
  tx = new complex<double>[numtx];
  for(int i = 0; i < numtx; i++){
    //double angle = 180+90;
    //double angle = 180+45+i*90.0/(numtx-1);
    double angle = 360.0/numtx*i;
    tx[i] = complex<double>(cos(angle/360*2*M_PI),sin(angle/360*2*M_PI));
  }
  //RECEIVERS (FARFIELD)
  rx = new complex<double>[numrx];
  for(int i = 0; i < numrx; i++){
    //double angle = 45+i*(90.0/(numrx-1));
    double angle = 360.0/numrx*i;
    rx[i] = complex<double>(cos(angle/360*2*M_PI),sin(angle/360*2*M_PI));
  }
  /*tx = new complex<double>[numtx];
  for(int m = 0; m < numtx; m++){
    double angle = -30+m*60.0/(numtx-1);
    tx[m] = complex<double>(sin(angle/360*2*M_PI),-cos(angle/360*2*M_PI));
    if(myid==0)printf("angle %d: %f\n",m,angle);
  }
  rx = new complex<double>[numrx];
  for(int m = 0; m < numrx; m++){
    rx[m] = complex<double> (-(numrx-1)*0.1/2+m*0.1,dim/2);
    if(myid==0)printf("pos %d: %e %e\n",m,rx[m].real(),rx[m].imag());
  }*/

  if(myid==0){
    printf("*******************************************************\n");
    printf("NUM. FREQ: %d\n",numfreq);
    printf("MY   FREQ: %f\n",scale);
    printf("LOW. FREQ: %f\n",scales[numfreq-1]);
    printf("NUM. TX: %d\n",numtx);
    printf("NUM. RX: %d\n",numrx);
    printf("RESOLUTION: %f\n",res);
    printf("OUTER MATRIX SIZE: %f G\n",numunk/1e6/1e6*numrx);
    printf("INNER MATRIX SIZE: %f G\n",numunk/1e6/1e6*numunk);
    printf("DIMENSION: %f WAVELENGTHS\n",dim);
    printf("NUM. UNKNOWNS: %d\n",numunk);
    printf("NUM. RHS: %d\n",numrx*numfreq*numtx);
  }

  //SETUP MLFMA
  if(myid==0)printf("SETUP MLFMA\n");
  setup_mlfma();
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("SETUP GPU\n");
  setup_gpu();
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("SETUP BICGS\n");
  setup_bicgs();
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("MLFMA SETUP DONE, ALLOCATIONS\n"); 

  inc = new complex<double>[txproc*numunk];
  tot = new complex<double>[txproc*numunk];
  //go = new complex<double>[numrx*numunk];
  //gh = new complex<double>[numunk*numrx];
  rhs = new complex<double>[txproc*numrx];
  mes = new complex<double>[txproc*numrx];
  mesd = new complex<double>[txproc*numrx];
  omes = new complex<double>[numunk];

  x = new complex<double>[numunk];
  o = new complex<double>[numunk];
  od = new complex<double>[numunk];
  y = new complex<double>[numunk];
  c = new complex<double>[numunk];
  g = new complex<double>[numunk];
  ot = new complex<double>[numunk];
  buff = new complex<double>[numunk];
  save = new complex<double>[numunk];

  //SETUP BORN
  if(myid==0)printf("SETUP BORN\n");
  setup_born();
  if(myid==0)printf("SETUP FINISHED\n");
  MPI_Barrier(MPI_COMM_WORLD);
  setupt = MPI_Wtime()-totalt;
  if(myid==0)printf("SETUP Time: %e\n",setupt);
  if(myid==0)errhist = fopen("hist.txt","w");

  int amount = numunk/numproc_mlfma;
  int addres = myid_mlfma*amount;
  //INITIAL GUESS (NO OBJECT)
  fill_n(o,numunk,complex<double>(0,0));
  memcpy(tot,inc,numunk*txproc*sizeof(complex<double>));
  #pragma omp parallel for
  for(int m = 0; m < numrx*txproc; m++)
    mesd[m] = -mes[m];
  /*//INITIAL GUESS READ
  FILE *initf = fopen("save/cg_20.bin","rb");
  fread(od,sizeof(complex<double>),numunk,initf);
  fclose(initf);
  #pragma omp parallel for
  for(int n = 0; n < numunk; n++)o[unkmap[n]]=od[n]*k0*k0;
  for(int i = 0; i < txproc; i++){
    fill_n(&tot[i*numunk],numunk,complex<double>(0,0));
    bicgs(&tot[i*numunk],o,&inc[i*numunk],false);
    for(int n = 0; n < numunk; n++)
      x[n] = tot[i*numunk+n]*o[n];
    aggregate(x,&rhs[i*numrx]);
    #pragma omp parallel for
    for(int m = 0; m < numrx; m++)
      mesd[i*numrx+m] = rhs[i*numrx+m] - mes[i*numrx+m];
  }*/

  //GRADIENT
  fill_n(&buff[addres],amount,complex<double>(0,0));
  for(int i = 0; i < txproc; i++){
    aggregateh(y,&mesd[i*numrx]);
    #pragma omp parallel for
    for(int n = 0; n < amount; n++)
      c[addres+n] = conj(o[addres+n]*k0*k0)*y[addres+n];
    fill_n(&x[addres],amount,complex<double>(0,0));
    #pragma omp parallel for
    for(int n = 0; n < amount; n++)
      g[addres+n] = o[addres+n]*k0*k0;
    bicgs(x,g,c,true);
    mlfmah(x,c);
    #pragma omp parallel for
    for(int n = 0; n < amount; n++)
      y[addres+n] = y[addres+n]+c[addres+n];
    #pragma omp parallel for
    for(int n = 0; n < amount; n++)
      buff[addres+n] = buff[addres+n] + conj(tot[i*numunk+addres+n])*y[addres+n];
    //REGULARIZE
    #pragma omp parallel for
    for(int n = 0; n < amount; n++)
      buff[addres+n] = buff[addres+n] + regular*regular*o[addres+n];
  }
  MPI_Allreduce(&buff[addres],&od[addres],amount,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_DBIM);

  //ERROR
  write_error(0);

  memcpy(&ot[addres],&od[addres],amount*sizeof(complex<double>));
  for(int iter = 0; iter <= 2; iter++){
    if(myid==0)printf("**************************** BORN %d *******************\n",iter);

    //DENOMINATOR
    for(int i = 0; i < txproc; i++){
      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        y[addres+n] = tot[i*numunk+addres+n]*ot[addres+n]*k0*k0;
      mlfma(y,c);
      fill_n(&x[addres],amount,complex<double>(0,0));
      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        g[addres+n] = o[addres+n]*k0*k0;
      bicgs(x,g,c,false);

  MPI_Gather(&c[addres],amount,MPI_DOUBLE_COMPLEX,buff,amount,MPI_DOUBLE_COMPLEX,0,MPI_COMM_MLFMA);
  if(myid==0){
    FILE *odf = fopen("od.bin","wb");
    fwrite(buff,sizeof(complex<double>),numunk,odf);
    fclose(odf);
  }

      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        y[addres+n] = y[addres+n] + o[addres+n]*k0*k0*x[addres+n];
      aggregate(y,&rhs[i*numrx]);
    }
    complex<double> temp1 = 0;
    complex<double> temp1tot;
    double temp2 = 0;
    double temp2tot;
    for(int i = 0; i < txproc; i++)
      for(int m = 0; m < numrx; m++){
        temp1 = temp1 + conj(mesd[i*numrx+m])*rhs[i*numrx+m];
        temp2 = temp2 + norm(rhs[i*numrx+m]);
      }
    MPI_Allreduce(&temp1,&temp1tot,1,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_DBIM);
    MPI_Allreduce(&temp2,&temp2tot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_DBIM);
    double alpha = temp1tot.real()/temp2tot;
    if(myid==0)printf("alpha: %e\n",alpha);
    //TAKE STEP
    #pragma omp parallel for
    for(int n = 0; n < amount; n++)
      o[addres+n] = o[addres+n] - alpha*ot[addres+n];

    //UPDATE GREEN'S FUNCTION
    for(int i = 0; i < txproc; i++){
      fill_n(&tot[i*numunk+addres],amount,complex<double>(0,0));
      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        g[addres+n] = o[addres+n]*k0*k0;
      bicgs(&tot[i*numunk],g,&inc[i*numunk],false);
      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        x[addres+n] = tot[i*numunk+addres+n]*o[addres+n]*k0*k0;
      aggregate(x,&rhs[i*numrx]);
      #pragma omp parallel for
      for(int m = 0; m < numrx; m++)
        mesd[i*numrx+m] = rhs[i*numrx+m] - mes[i*numrx+m];
    }
    memcpy(&save[addres],&od[addres],amount*sizeof(complex<double>));
    //GRADIENT
    fill_n(&buff[addres],amount,complex<double>(0,0));
    for(int i = 0; i < txproc; i++){
      aggregateh(y,&mesd[i*numrx]);
      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        c[addres+n] = conj(o[addres+n]*k0*k0)*y[addres+n];
      fill_n(&x[addres],amount,complex<double>(0,0));
      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        g[addres+n] = o[addres+n]*k0*k0;
      bicgs(x,g,c,true);
      mlfmah(x,c);
      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        y[addres+n] = y[addres+n]+c[addres+n];
      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        buff[addres+n] = buff[addres+n] + conj(tot[i*numunk+addres+n])*y[addres+n];
      //REGULARIZE
      #pragma omp parallel for
      for(int n = 0; n < amount; n++)
        buff[addres+n] = buff[addres+n] + regular*regular*o[addres+n];
    }
    MPI_Allreduce(&buff[addres],&od[addres],amount,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_DBIM);

    //ERROR
    write_error(iter);

    /*//BETA (STEEPEST-DESCENT)
    double beta = 0;
    if(myid==0)printf("SD beta: %e\n",beta);*/
    /*//BETA (FLETCHER-REEVES)
    double norm1 = 0;
    double norm1tot = 0;
    double norm2 = 0;
    double norm2tot = 0;
    for(int n = 0; n < amount; n++){
      norm1 = norm1 + norm(od[addres+n]);
      norm2 = norm2 + norm(save[addres+n]);
    }
    MPI_Allreduce(&norm1,&norm1tot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_MLFMA);
    MPI_Allreduce(&norm2,&norm2tot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_MLFMA);
    double beta = norm1tot/norm2tot;
    if(myid==0)printf("FR beta: %e\n",beta);*/
    //BETA (POLAK-RIBIERE)
    complex<double> norm1 = 0;
    complex<double> norm1tot = 0;
    double norm2 = 0;
    double norm2tot = 0;
    for(int n = 0; n < numunk; n++){
      norm1 = norm1 + od[n]*conj(od[n]-save[n]);
      norm2 = norm2 + norm(save[n]);
    }
    MPI_Allreduce(&norm1,&norm1tot,1,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_MLFMA);
    MPI_Allreduce(&norm2,&norm2tot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_MLFMA);
    complex<double> beta = norm1tot/norm2tot;
    if(myid==0)printf("PR beta: %e %e\n",beta.real(),beta.imag());

    #pragma omp parallel for
    for(int n = 0; n < amount; n++)
      ot[addres+n] = od[addres+n] + beta*ot[addres+n];
  }
  if(myid == 0)fclose(errhist);

  int matvecintot = 0;
  int matvecoutot = 0;
  MPI_Reduce(&matvecin,&matvecintot,1,MPI_INT,MPI_SUM,0,MPI_COMM_DBIM);
  MPI_Reduce(&matvecout,&matvecoutot,1,MPI_INT,MPI_SUM,0,MPI_COMM_DBIM);

  if(myid==0)printf("OUTER MATVEC: %d INNER MATVEC: %d\n",matvecoutot,matvecintot);

  MPI_Barrier(MPI_COMM_WORLD);
  totalt = MPI_Wtime()-totalt;
  solutt = totalt-setupt;
  if(myid==0){
    printf("TOTAL TIME : %e\n",totalt);
    printf("SETUP TIME : %e\n",setupt);
    printf("SOLUT TIME : %e\n",solutt);
    printf("BICGS TIME : %e\n",bicgst);
    printf("INNER TIME : %e\n",innert);
    printf("OUTER TIME : %e\n",outert);
  }
  MPI_Finalize();
}
