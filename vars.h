#include <cstdio>
#include <cmath>
#include <complex>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#include <cuda.h>
#include <cstring>

#include <sys/time.h>

using namespace std;

void mlfma(complex<double>*,complex<double>*);
void mlfmah(complex<double>*,complex<double>*);
void aggregate(complex<double>*,complex<double>*);
void aggregateh(complex<double>*,complex<double>*);
void setup_mlfma();
void setup_born();
void write_error(int);

//SOLVER ROUTINES
void setup_bicgs();
void matvec(complex<double>*,complex<double>*,complex<double>*);
void matvech(complex<double>*,complex<double>*,complex<double>*);
void bicgs(complex<double>*,complex<double>*,complex<double>*,bool);

//INTEGRATE ROUTINES
complex<double> integrate(complex<double>,complex<double>);
complex<double> integrate_multi(complex<double>,complex<double>,int);
complex<double> integrate_local(complex<double>,complex<double>,int);
complex<double> hn(int, double);
//FARFIELD
void farfield(complex<double>*);

//GPU ROUTINES
void setup_gpu();
void mlfma_cpu(complex<double>*,complex<double>*);
