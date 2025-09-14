#pragma once
//Linear Solution Routines
//Mert Hidayetoglu, Oct 2015
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
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuComplex.h>
#include <random>
#include "kernels.cuh"


using namespace std;

#define M_PI 3.141592653589793

struct timeReport {
	float milliseccondsCopy = 0.0;
	float milliseccondsKernel = 0.0;
	int numMLFMAcalls = 0;
	float totalTime() { return milliseccondsCopy + milliseccondsKernel; }
};

class FMMFourierTreeGPU
{
public: FMMFourierTreeGPU(int N, int PointsPerBox);
public: ~FMMFourierTreeGPU();

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

	int nummatvec;

	double toli = 1e-4;
	int iti = 300; //bicgs max iterations
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
	int numproc_mlfma;
	int itfor;
	int itinv;
	int matvecin = 0;
	int matvecout = 0;

	double setupt;
	double solutt;
	double innert=0;
	double outert;
	double totalt;
	double bicgst;



	#pragma endregion

	#pragma region GPU variables
	cuDoubleComplex* coeff_d;
	cuDoubleComplex* basis_d;
	cuDoubleComplex** aggmulti_d;
	cuDoubleComplex** agglocal_d;
	double** interp_d;
	double** anterp_d;
	int** intind_d;
	int** antind_d;
	cuDoubleComplex** shiftmul_d;
	cuDoubleComplex** shiftloc_d;
	cuDoubleComplex** trans_d;
	int** traid_d;
	int** clusfar_d;
	int* clusnear_d;
	cuDoubleComplex* near_d;

	int** sendmap_d;
	int** recvmap_d;
	int** procmap_d;
	int** clusmap_d;
	int** procint_d;
	int* clcount_d;
	cuDoubleComplex** sendbuff_d;
	cuDoubleComplex** recvbuff_d;
	complex<double>** sendbuff_h;
	complex<double>** recvbuff_h;
	cudaStream_t commstr;
	cudaStream_t kerrstr;

	complex<double>* x_h;
	complex<double>* r_h;
	cuDoubleComplex* x_d;
	cuDoubleComplex* r_d;
	double gpumem;


	cuDoubleComplex* cbuff;
	complex<double>* cbuff_h;
	double* dbuff;
	double* dbuff_h;

	cuDoubleComplex* p;
	cuDoubleComplex* v;
	cuDoubleComplex* s;
	cuDoubleComplex* t;
	cuDoubleComplex* r_tld;

	cuDoubleComplex* o_d;
	cuDoubleComplex* b_d;
	#pragma endregion

	public: int bornIteratoins = 3; //control DBIM iterations


	#pragma region functions
	void mlfma_gpu(cuDoubleComplex*, cuDoubleComplex*);
	void mlfma_gpu_farField(cuDoubleComplex*, cuDoubleComplex*);
	void mlfma_gpu_nearField(cuDoubleComplex*, cuDoubleComplex*);
	void mlfma(complex<double>* x, complex<double>* r);
	void mlfmah(complex<double>* x, complex<double>* r);
	void aggregate(complex<double>* x, complex<double>* r);
	void aggregateh(complex<double>* x, complex<double>* r);
	void farfield(complex<double>* x);
	double norm2(cuDoubleComplex* a);
	complex<double>inner(cuDoubleComplex* a, cuDoubleComplex* b);
	void matvec(cuDoubleComplex* x, cuDoubleComplex* o, cuDoubleComplex* r, bool ishermitian);
	complex<double>hn(int order, double dist);
	double integ(double x, double y);
	complex<double> integrate(complex<double> post, complex<double> posb);
	complex<double> integrate_multi(complex<double> center, complex<double> posb, int order);
	complex<double>	integrate_local(complex<double> center, complex<double> post, int order);
	void bicgs(complex<double>* x, complex<double>* o, complex<double>* b, bool ishermitian, bool verbos);
	int MoM(complex<double>* measuredField = NULL);
	void direct(complex<double>* x, complex<double>* r);
	#pragma endregion

	#pragma region setup
	void setup_mlfma();
	void setup_interp();
	void setup_gpu();
	void setup_bicgs();
	void setup_born();
	#pragma endregion

	//CUDA kernels moved to other file

	#pragma region check redundancy
	cuDoubleComplex* near_red; //redundant nearField data
	float totalCopyTime = 0.0, 
		totalDuplicationTime = 0.0,
		totalComputationTime = 0.0,
		totalMlfmaTime = 0.0,
		totalFarFieldTime = 0.0,
		totalNearFieldTime = 0.0;
	vector<vector<complex<double>>> ot_i; //vector of finalObjects ot, at the end of each born iteration
	vector<complex<double>> generatedField;

	void mlfma_redundant(complex<double>* x, complex<double>* r, int redundancyFactor);
	float nearfield_old(complex<double>* x, complex<double>* r);
	float nearfield_modified(complex<double>* x, complex<double>* r);
	void nearfield_redundant(complex<double>* x, complex<double>* r, int redundancyFactor);
	void nearfield_redundant(cuDoubleComplex* x_, cuDoubleComplex* r_, int redundancyFactor);
	void mlfma_gpu_redundant(cuDoubleComplex* x_, cuDoubleComplex* r_, int redundancyFactor);
	void duplicateData(int redundancyFactor);////allocate memory if required, and duplicate data
	int MoM_redundancy(int redundancyFactor, complex<double>* measuredField = NULL);
	void bicgs_redundancy(complex<double>* x, complex<double>* o, complex<double>* b, bool ishermitian, bool verbos, int redundancyFactor);
	void matvec_redundancy(cuDoubleComplex* x, cuDoubleComplex* o, cuDoubleComplex* r, bool ishermitian, int redundancyFactor);
	void generateRandomMeasuredField();
#pragma endregion
};
