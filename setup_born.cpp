#include "vars.h"

double randf(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void setup_born(){

  extern MPI_Comm MPI_COMM_MLFMA;
  extern MPI_Comm MPI_COMM_DBIM;
  extern double k0;
  extern double res;
  extern int numrx;
  extern int txproc;
  extern int mytx;
  extern int numunk;
  extern complex<double> *rx;
  extern complex<double> *tx;
  extern complex<double> *go;
  extern complex<double> *gh;
  extern complex<double> *inc;
  extern complex<double> *pos;
  extern int *unkmap;

  extern complex<double> *omes;
  extern double mesnorm;
  extern double objnorm;
  extern complex<double> *mes;
  extern complex<double> *x;
  extern complex<double> *o;
  extern complex<double> *c;
  extern complex<double> *g;
  extern complex<double> *buff;

  char strtemp[80];
  //extern int myfreq;
  //extern int numtx;
  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  int numproc_mlfma;
  int myid_mlfma;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc_mlfma);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid_mlfma);
  int numproc_dbim;
  int myid_dbim;
  MPI_Comm_size(MPI_COMM_DBIM,&numproc_dbim);
  MPI_Comm_rank(MPI_COMM_DBIM,&myid_dbim);

  /*#pragma omp parallel for
  for(int m = 0; m < numrx; m++)
    for(int n = 0; n < numunk; n++)
      //go[m*numunk+n] = exp(complex<double>(0,-k0*(rx[m].real()*pos[n].real()+rx[m].imag()*pos[n].imag())))*res*res;
      go[m*numunk+n] = integrate(rx[m],pos[n]);
  #pragma omp parallel for
  for(int n = 0; n < numunk; n++)
    for(int m = 0; m < numrx; m++)
      gh[n*numrx+m]=conj(go[m*numunk+n]);*/

  for(int i = 0; i < txproc; i++)
    #pragma omp parallel for
    for(int n = 0; n < numunk; n++)
      inc[i*numunk+n]=exp(complex<double>(0,k0*(tx[mytx+i].real()*pos[n].real()+tx[mytx+i].imag()*pos[n].imag())));
    /*{inc[i*numunk+n]=0;
      for(int m = 0; m < numrx; m++){
        complex<double> ric = rx[m]-rx[0];
        complex<double> tth = tx[mytx+i];
        if(tx[mytx+i].real()>0)ric=rx[m]-rx[numrx-1];
        inc[i*numunk+n]=inc[i*numunk+n]+integrate(rx[m],pos[n])/res/res*exp(complex<double>(0,k0*(tth.real()*ric.real()+tth.imag()*ric.imag())));
      }
    }
  for(int i = 0; i < txproc; i++){
    for(int n = 0; n < numunk; n++)
      buff[n] = inc[i*numunk+unkmap[n]];
    sprintf(strtemp,"inc_%d_of%d.bin",i,myid_dbim);
    FILE *incf = fopen(strtemp,"wb");
    fwrite(buff,sizeof(complex<double>),numunk,incf);
    fclose(incf);
  }*/
    

  //CALCULATE RHS
  /*fill_n(o,numunk,complex<double>(0,0));
  #pragma omp parallel for
  for(int n = 0; n < numunk; n++){
    //if(abs(pos[n]-complex<double>(0,6))<3)
    //    o[n] = eps*cos(abs(pos[n]-complex<double>(0,6))/(3/M_PI*2));
    //if(abs(pos[n])<12)o[n]=0.1;
    //if(abs(pos[n])<8)
    //  if(abs(pos[n])>3)
    //    o[n] = 0.05;
    if(pos[n].real()<9 && pos[n].real()>-9 && pos[n].imag()<9 && pos[n].imag()>-9)
        o[n] = 0.1;
    if(pos[n].real()<8 && pos[n].real()>-8 && pos[n].imag()<8 && pos[n].imag()>-8)
        o[n] = 0.0;
      if(abs(pos[n])<3)
        o[n] = 0.1;
    //if(abs(pos[n])<res)
    //  o[n] = 1;
  }*/
  if(myid==0){
    printf("READ FROM FILE\n");
    FILE *inputf = fopen("../phantoms/phantom_256.txt","r");
    for(int n = 0; n < numunk; n++){
      float val = 0;
      fscanf(inputf,"%f\n",&val);
      o[unkmap[n]] = complex<double>(val,0)*complex<double>(0.1,0);
    }
    fclose(inputf);
    printf("DONE\n");
  }
  MPI_Bcast(o,numunk,MPI_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
  if(myid == 0){
    for(int n = 0; n < numunk; n++)
      buff[n] = o[unkmap[n]];
    FILE *of = fopen("cg_ref.bin","wb");
    fwrite(buff,sizeof(complex<double>),numunk,of);
    fclose(of);
  }

  /*fill_n(mes,txproc*numrx,complex<double>(0,0));
  fill_n(buff,numunk,complex<double>(0,0));
  mes[0] = 1;
  aggregateh(buff,mes);
  int amount = numunk/numproc_mlfma;
  int addres = myid_mlfma*amount;
  MPI_Allgather(&buff[addres],amount,MPI_DOUBLE_COMPLEX,x,amount,MPI_DOUBLE_COMPLEX,MPI_COMM_MLFMA);
  if(myid == 0){
    printf("MFLMA ID: %d MLFMA PROC %d\n",myid_mlfma,numproc_mlfma);
    for(int n = 0; n < numunk; n++)
      buff[n] = x[unkmap[n]];
    FILE *of = fopen("field.bin","wb");
    fwrite(buff,sizeof(complex<double>),numunk,of);
    fclose(of);
  }*/

  for(int i = 0; i < txproc; i++){
    fill_n(x,numunk,complex<double>(0,0));
    #pragma omp parallel for
    for(int n = 0; n < numunk; n++)
      g[n] = o[n]*k0*k0;
    bicgs(x,g,&inc[i*numunk],false);
    #pragma omp parallel for
    for(int n = 0; n < numunk; n++)
      x[n] = x[n]*g[n];
    aggregate(x,&mes[i*numrx]);
  }
  /*complex<double> *mesall = new complex<double>[txproc*numrx*numproc_dbim];
  MPI_Gather(mes,txproc*numrx,MPI_DOUBLE_COMPLEX,mesall,txproc*numrx,MPI_DOUBLE_COMPLEX,0,MPI_COMM_DBIM);
  if(myid==0){
    printf("rhs: %d\n",txproc*numrx*numproc_dbim);
    FILE *mesf = fopen("mes.bin","wb");
    fwrite(mesall,sizeof(complex<double>),txproc*numrx*numproc_dbim,mesf);
    fclose(mesf);
  }*/
  objnorm = 0;
  memcpy(omes,o,numunk*sizeof(complex<double>));
  for(int n = 0; n < numunk; n++)
    objnorm = objnorm + norm(omes[n]);
  double err = 0;
  for(int i = 0; i < txproc; i++){
    for(int m = 0; m < numrx; m++)
      err = err + norm(mes[i*numrx+m]);
  }
  MPI_Allreduce(&err,&mesnorm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_DBIM);
  //printf("MLFMA ID: %d DBIM ID: %d MEASUREMENT NORM: %e (SCALE: %e)\n",myid_mlfma,myid_dbim,err,k0/(2*M_PI));
}

void write_error(int iter){

  extern int txproc;
  extern int numrx;
  extern int numunk;
  extern double regular;
  extern double objnorm;
  extern double mesnorm;
  extern int *unkmap;
  extern FILE *errhist;

  extern complex<double> *o;
  extern complex<double> *y;
  extern complex<double> *od;
  extern complex<double> *mesd;
  extern complex<double> *omes;
  extern complex<double> *buff;

  extern MPI_Comm MPI_COMM_MLFMA;
  extern MPI_Comm MPI_COMM_DBIM;

  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  int numproc_mlfma;
  int myid_mlfma;
  MPI_Comm_size(MPI_COMM_MLFMA,&numproc_mlfma);
  MPI_Comm_rank(MPI_COMM_MLFMA,&myid_mlfma);
  int numproc_dbim;
  int myid_dbim;
  MPI_Comm_size(MPI_COMM_DBIM,&numproc_dbim);
  MPI_Comm_rank(MPI_COMM_DBIM,&myid_dbim);

  int amount = numunk/numproc_mlfma;
  int addres = myid_mlfma*amount;
  MPI_Allgather(&o[addres],amount,MPI_DOUBLE_COMPLEX,y,amount,MPI_DOUBLE_COMPLEX,MPI_COMM_MLFMA);
  memcpy(o,y,numunk*sizeof(complex<double>));
  if(myid==0){
    char strtemp[80];
    sprintf(strtemp,"cg_%d.bin",iter);
    #pragma omp parallel for
    for(int n = 0; n < numunk; n++)
      buff[n] = o[unkmap[n]];
    FILE *file = fopen(strtemp,"wb");
    fwrite(buff,sizeof(complex<double>),numunk,file);
    fclose(file);
  }
  double err = 0;
  double errd = 0;
  double errtot;
  double errdtot;
  for(int i = 0; i < txproc; i++)
    for(int m = 0; m < numrx; m++)
      err = err + norm(mesd[i*numrx+m]);
  MPI_Reduce(&err,&errtot,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_DBIM);
  for(int n = 0; n < amount; n++)
    errd = errd + norm(od[addres+n]);
  MPI_Reduce(&errd,&errdtot,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_MLFMA);
  double obj = 0;
  double objn = 0;
  double errm = 0;
  for(int n = 0; n < numunk; n++){
    obj = obj + norm(regular*o[n]);
    objn = objn + norm(o[n]);
    errm = errm + norm(o[n]-omes[n]);
  }
  if(myid==0){
    fprintf(errhist,"%e %e %e %e %e\n",errtot/mesnorm,errm/objnorm,obj/mesnorm,errdtot,objn);
    printf("MEASUREMENT COST: %e OBJECT ERROR: %e OBJECT COST: %e GRADIENT NORM: %e OBJ NORM: %e\n",errtot/mesnorm,errm/objnorm,obj/mesnorm,errdtot,objn);
  }
}

