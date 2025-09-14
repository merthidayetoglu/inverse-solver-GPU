#pragma once
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <cmath>
#include <complex>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstring>
#include <random>
#include <time.h>
using namespace std;

#define M_PI 3.14

class FMMFourierTree
{
public: FMMFourierTree(int N, int PointsPerBox);

#pragma region algorithm parameters
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

	double dim;
	int numunk;

	int nummatvec;//number of MLFMA calls

	double toli = 1e-4;
	int iti = 300;
	double tolo = 1e-4;
	int ito = 300;

	int beta = 5;
	int ninter = 30;

	int* numclus;
	double* clusize;
	int* numsamp;
	int* numterm;

	complex<double>** clusc;
	int** clusnear;
	int** clusfar;
	int** cluspar;
	int** cluschi;
	int* unkpar;

	complex<double>* near;
	complex<double>* coeff_multi;
	complex<double>* basis_local;

	double** interp;
	double** anterp;
	int** intind;
	int** antind;
	complex<double>** shiftmul;
	complex<double>** shiftloc;
	complex<double>** trans;
	int** traid;
	complex<double>** aggmulti;
	complex<double>** agglocal;
	complex<double>** temp;
	//HIGHEST-LEVEL INTERPOLATORS
	double* interph;
	double* anterph;
	int* intindh;
	int* antindh;

	int** sendmap;
	int** recvmap;
	int** procmap;
	int** clusmap;
	int** procint;
	int* clcount;
	complex<double>** sendbuff;
	complex<double>** recvbuff;
#pragma endregion

#pragma region variables
	complex<double>* rx;
	complex<double>* tx;
	complex<double>* go;
	complex<double>* gh;
	double* scales;
	double scale;
	double regular = 0.0;

	complex<double>* o;
	complex<double>* od;
	complex<double>* inc;
	complex<double>* tot;
	complex<double>* pos;
	int* unkmap;

	double mesnorm;
	double objnorm;
	complex<double>* omes;
	complex<double>* mes;
	complex<double>* mesd;
	complex<double>* rhs;
	complex<double>* x;
	complex<double>* y;
	complex<double>* c;
	complex<double>* g;
	complex<double>* ot;
	complex<double>* buff;
	complex<double>* save;

	FILE* errhist;
	FILE* ithist;
	FILE* file;
	char* chartemp;
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

	complex<double>* p;
	complex<double>* v;
	complex<double>* s;
	complex<double>* t;
	complex<double>* r;
	complex<double>* r_tld;
	complex<double>* buffer;
#pragma endregion

#pragma region CPU functions
	void mlfma(complex<double>*, complex<double>*);
	void direct(complex<double>* x, complex<double>* r);
	void mlfmah(complex<double>*, complex<double>*);
	void aggregate(complex<double>*, complex<double>*);
	void aggregateh(complex<double>*, complex<double>*);
	double integ(double x, double y);
	double norm2(complex<double>* a);
	complex<double>inner(complex<double>* a, complex<double>* b);
	void saxpy(complex<double>* a, complex<double>* b, complex<double>* c, complex<double> alpha);
	void matvec(complex<double>* x, complex<double>* o, complex<double>* b, bool ishermitian);
	void setup_mlfma();
	double randf(double fMin, double fMax);
	void setup_born();

	//SOLVER ROUTINES
	void setup_bicgs();
	void matvec(complex<double>*, complex<double>*, complex<double>*);
	void matvech(complex<double>*, complex<double>*, complex<double>*);
	void bicgs(complex<double>*, complex<double>*, complex<double>*, bool);
	int MoM();

	//INTEGRATE ROUTINES
	complex<double> integrate(complex<double>, complex<double>);
	complex<double> integrate_multi(complex<double>, complex<double>, int);
	complex<double> integrate_local(complex<double>, complex<double>, int);
	complex<double> hn(int, double);
	//FARFIELD
	void farfield(complex<double>*);

	//GPU ROUTINES
	void setup_gpu();
	void mlfma_cpu(complex<double>*, complex<double>*);
	void setup_interp();
	void aggregatehsetup_mlfma();
	void aggregatehsetup_interp();
#pragma endregion

	vector<complex<double>> generatedField;

	void generateRandomMeasuredField();
	void matvec_mvp(complex<double>* x, complex<double>* o, complex<double>* b);
	complex<double>* __z;
	void getEmpedanceMatrix();
	void bicgs_mvp(complex<double>*, complex<double>*, complex<double>*, bool);
};

