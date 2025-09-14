#include "FMMFourierTree.h"

FMMFourierTree::FMMFourierTree(int N, int PointsPerBox)
{
    //level = L;
    box = sqrt(PointsPerBox);
    level = log(N / (box*box)) / log(4) + 1;

    int numproc = 1;
    int myid = 0;
    //MPI_Init(&argc, &argv);
    //MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    //MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int numproc_mlfma = 1;
    int myid_mlfma = 0;

    //MPI_Barrier(MPI_COMM_WORLD);
    //totalt = MPI_Wtime();

    //chartemp = getenv("NUMFREQ");
    numfreq = 1;// atoi(chartemp);
    //chartemp = getenv("NUMTX");
    numtx = 1024;// atoi(chartemp);
    //chartemp = getenv("TXPROC");
    txproc = 1;// atoi(chartemp);
    //chartemp = getenv("NUMRX");
    numrx = 1024;// atoi(chartemp);
    //chartemp = getenv("MLFMA");
    numproc_mlfma = 1;
    //chartemp = getenv("MINFREQ");
    double minfreq = 0.25;// atof(chartemp);
    //chartemp = getenv("MAXFREQ");
    double maxfreq = 1;// atof(chartemp);


    int myid_dbim = 0;
    mytx = (myid_dbim * txproc) % numtx;
    myfreq = myid_dbim / (numtx / txproc);

    //if (myid == 0)
    //FREQUENCIES
    scales = new double[numfreq];
#pragma omp parallel
    for (int i = 0; i < numfreq; i++) {
        scale = maxfreq - (maxfreq - minfreq) / (numfreq - 1) * i;
        if (numfreq == 1)scale = maxfreq;
        scales[i] = scale;
    }
    k0 = 2 * M_PI * scales[myfreq];
    dim = res * box * pow(2, level - 1);
    numunk = pow(dim / res, 2);
    printf("MLFMA with N=%d, Level=%d\n", numunk, level);

    //TRANSMITTERS (FARFIELD)
    tx = new complex<double>[numtx];
    for (int i = 0; i < numtx; i++) {
        //double angle = 180+90;
        //double angle = 180+45+i*90.0/(numtx-1);
        double angle = 360.0 / numtx * i;
        tx[i] = complex<double>(cos(angle / 360 * 2 * M_PI), sin(angle / 360 * 2 * M_PI));
    }
    //RECEIVERS (FARFIELD)
    rx = new complex<double>[numrx];
    for (int i = 0; i < numrx; i++) {
        //double angle = 45+i*(90.0/(numrx-1));
        double angle = 360.0 / numrx * i;
        rx[i] = complex<double>(cos(angle / 360 * 2 * M_PI), sin(angle / 360 * 2 * M_PI));
    }

    //SETUP MLFMA
    printf("SETUP MLFMA\n");
    setup_mlfma();
    printf("SETUP BICGS\n");
    setup_bicgs();
    

    inc = new complex<double>[txproc * numunk];
    tot = new complex<double>[txproc * numunk];
    //go = new complex<double>[numrx*numunk];
    //gh = new complex<double>[numunk*numrx];
    rhs = new complex<double>[txproc * numrx];
    mes = new complex<double>[txproc * numrx];
    mesd = new complex<double>[txproc * numrx];
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

    ////SETUP BORN
    //printf("SETUP BORN...\n");
    //setup_born();
    //printf("SETUP FINISHED\n");
}

#pragma region utils
double FMMFourierTree::randf(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
double FMMFourierTree::norm2(complex<double>* a) {
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    double ret = 0;
    for (int m = myid * numunk / numproc; m < (myid + 1) * numunk / numproc; m++)
        ret = ret + norm(a[m]);
    //double rettot;
    //MPI_Allreduce(&ret, &rettot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_MLFMA);
    //return rettot;
    return ret;
}
void FMMFourierTree::saxpy(complex<double>* a, complex<double>* b, complex<double>* c, complex<double> alpha) {//c = a + alpha*b
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
#pragma omp parallel for
    for (int n = myid * numunk / numproc; n < (myid + 1) * numunk / numproc; n++)
        c[n] = a[n] + alpha * b[n];
}
complex<double> FMMFourierTree::inner(complex<double>* a, complex<double>* b) {
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    complex<double> red = 0;
    for (int m = myid * numunk / numproc; m < (myid + 1) * numunk / numproc; m++)
        red = red + conj(a[m]) * b[m];
    //complex<double> redtot;
    //MPI_Allreduce(&red, &redtot, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_MLFMA);
    //return redtot;
    return red;
}
#pragma endregion

#pragma region solve CPU
void FMMFourierTree::mlfmah(complex<double>* x, complex<double>* r) {
    //extern MPI_Comm MPI_COMM_MLFMA;
    //extern int numunk;
    int myid = 0;
    int numproc = 1;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
#pragma omp parallel for
    for (int n = myid * numunk / numproc; n < (myid + 1) * numunk / numproc; n++)
        x[n] = conj(x[n]);
    mlfma(x, r);
#pragma omp parallel for
    for (int n = myid * numunk / numproc; n < (myid + 1) * numunk / numproc; n++) {
        r[n] = conj(r[n]);
        x[n] = conj(x[n]);
    }
}
void FMMFourierTree::mlfma(complex<double>* x, complex<double>* r) {

    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int numproc_g = 1;
    int myid_g = 0;
    //MPI_Comm_size(MPI_COMM_WORLD, &numproc_g);
    //MPI_Comm_rank(MPI_COMM_WORLD, &myid_g);

    matvecin++;

    //INITIALIZATION
    for (int i = 0; i < level; i++)
#pragma omp parallel for
        for (int clusm = myid * numclus[i] / numproc; clusm < (myid + 1) * numclus[i] / numproc; clusm++)
            fill_n(&agglocal[i][clusm * numsamp[i]], numsamp[i], complex<double>(0, 0));
    //LOWEST-LEVEL AGGREGATION
#pragma omp parallel for
    for (int clusm = myid * numclus[level - 1] / numproc; clusm < (myid + 1) * numclus[level - 1] / numproc; clusm++) {
        int indmulti = clusm * numsamp[level - 1];
        int unk = clusm * box * box;
        for (int k = 0; k < numsamp[level - 1]; k++) {
            complex<double> reduce = 0;
            int indbasis = k * box * box;
            for (int n = 0; n < box * box; n++)
                reduce = reduce + coeff_multi[indbasis + n] * x[unk + n];
            aggmulti[level - 1][indmulti + k] = reduce;
        }
    }
    //HIGHER-LEVEL AGGREGATIONS
    for (int i = level - 2; i > 1; i--) {
#pragma omp parallel for
        for (int clusm = myid * numclus[i] / numproc; clusm < (myid + 1) * numclus[i] / numproc; clusm++) {
            int indm = clusm * numsamp[i];
            for (int km = 0; km < numsamp[i]; km++) {
                complex<double> temp1 = 0;
                for (int cn = 0; cn < 4; cn++) {
                    int indn = (clusm * 4 + cn) * numsamp[i + 1];
                    complex<double> reduce = 0;
                    //INTERPOLATE
                    for (int k = 0; k < ninter; k++)
                        reduce = reduce + interp[i][km * ninter + k] * aggmulti[i + 1][indn + intind[i][km * ninter + k]];
                    //SHIFT
                    temp1 = temp1 + reduce * shiftmul[i][cn * numsamp[i] + km];
                }
                aggmulti[i][indm + km] = temp1;
            }
        }
    }
    for (int i = 2; i < level; i++)
#pragma omp parallel for
        for (int cl = 0; cl < clcount[i]; cl++)
            memcpy(&sendbuff[i][cl * numsamp[i]], &aggmulti[i][sendmap[i][cl] * numsamp[i]], numsamp[i] * sizeof(complex<double>));
    //FARFIELD COMMUNICATION
    for (int i = 2; i < level; i++) {
        for (int p = 0; p < numproc; p++) {
            int sib = procmap[i][p];
            if (sib != -1) {
                complex<double>* send = &sendbuff[i][procint[i][p] * numsamp[i]];
                complex<double>* recv = &recvbuff[i][procint[i][p] * numsamp[i]];
                int amount = clusmap[i][p] * numsamp[i];
                //MPI_Sendrecv(send, amount, MPI_DOUBLE_COMPLEX, sib, 0, recv, amount, MPI_DOUBLE_COMPLEX, sib, 0, MPI_COMM_MLFMA, MPI_STATUS_IGNORE);
            }
        }
    }
    for (int i = 2; i < level; i++)
#pragma omp parallel for
        for (int cl = 0; cl < clcount[i]; cl++)
            memcpy(&aggmulti[i][recvmap[i][cl] * numsamp[i]], &recvbuff[i][cl * numsamp[i]], numsamp[i] * sizeof(complex<double>));
    //TRANSLATION
    for (int i = 2; i < level; i++) {
#pragma omp parallel for
        for (int clusm = myid * numclus[i] / numproc; clusm < (myid + 1) * numclus[i] / numproc; clusm++) {
            int indlocal = clusm * numsamp[i];
            for (int cn = 0; cn < 27; cn++) {
                int clusn = clusfar[i][clusm * 27 + cn];
                if (clusn != -1) {
                    int indmulti = clusn * numsamp[i];
                    int index = traid[i][clusm * 27 + cn] * numsamp[i];
                    for (int k = 0; k < numsamp[i]; k++)
                        agglocal[i][indlocal + k] = agglocal[i][indlocal + k] + trans[i][index + k] * aggmulti[i][indmulti + k];
                }
            }
        }
    }
    //HIGHER-LEVEL DISAGGREGATIONS
    for (int i = 2; i < level - 1; i++) {
#pragma omp parallel for
        for (int clusn = myid * numclus[i] / numproc; clusn < (myid + 1) * numclus[i] / numproc; clusn++) {
            int indn = clusn * numsamp[i];
            int mythread = omp_get_thread_num();
            for (int cm = 0; cm < 4; cm++) {
                int indm = (clusn * 4 + cm) * numsamp[i + 1];
                //SHIFT
                for (int kn = 0; kn < numsamp[i]; kn++)
                    temp[mythread][kn] = shiftloc[i][cm * numsamp[i] + kn] * agglocal[i][indn + kn];
                //ANTERPOLATE
                for (int km = 0; km < numsamp[i + 1]; km++) {
                    complex<double> reduce = 0;
                    for (int k = 0; k < 2 * ninter; k++)
                        reduce = reduce + anterp[i + 1][km * 2 * ninter + k] * temp[mythread][antind[i + 1][km * 2 * ninter + k]];
                    agglocal[i + 1][indm + km] = agglocal[i + 1][indm + km] + reduce;
                }
            }
        }
    }
    //LOWEST-LEVEL DISAGGREGATION
#pragma omp parallel for
    for (int clusm = myid * numclus[level - 1] / numproc; clusm < (myid + 1) * numclus[level - 1] / numproc; clusm++) {
        int unk = clusm * box * box;
        int indlocal = clusm * numsamp[level - 1];
        for (int n = 0; n < box * box; n++) {
            complex<double> reduce = 0;
            int indbasis = n * numsamp[level - 1];
            for (int k = 0; k < numsamp[level - 1]; k++)
                reduce = reduce + basis_local[indbasis + k] * agglocal[level - 1][indlocal + k];
            r[unk + n] = reduce;
        }
    }
#pragma omp parallel for
    for (int cl = 0; cl < clcount[level]; cl++)
        memcpy(&sendbuff[level][cl * box * box], &x[sendmap[level][cl] * box * box], box * box * sizeof(complex<double>));
    
#pragma omp parallel for
    for (int cl = 0; cl < clcount[level]; cl++)
        memcpy(&x[recvmap[level][cl] * box * box], &recvbuff[level][cl * box * box], box * box * sizeof(complex<double>));
    //NEARFIELD
#pragma omp parallel for
    for (int clusm = myid * numclus[level - 1] / numproc; clusm < (myid + 1) * numclus[level - 1] / numproc; clusm++) {
        int testing = clusm * box * box;
        for (int m = 0; m < box * box; m++) {
            complex<double> reduce = 0;
            for (int cn = 0; cn < 9; cn++) {
                int clusn = clusnear[level - 1][clusm * 9 + cn];
                if (clusn != -1) {
                    int basis = clusn * box * box;
                    int indbox = cn * box * box * box * box + m * box * box;
                    for (int n = 0; n < box * box; n++)
                        reduce = reduce + x[basis + n] * near[indbox + n];
                }
            }
            r[testing + m] = r[testing + m] + reduce;
        }
    }
}
void FMMFourierTree::direct(complex<double>* x, complex<double>* r)
{
    int prog100 = numclus[level - 1] / 100;
    for (int clusm = 0; clusm < numclus[level - 1]; clusm++) 
    {
        if (clusm % prog100 == 0)
            printf("\rprogress [%d%%]", clusm / prog100);

        int testing = clusm * box * box;
        for (int m = 0; m < box * box; m++)
        {
            complex<double> reduce = 0;
            for (int clusn = 0; clusn < numclus[level - 1]; clusn++)
            {
                int basis = clusn * box * box;
                for (int n = 0; n < box * box; n++)
                {
                    reduce = reduce + x[basis + n] * integrate(pos[testing + m], pos[basis + n]);
                }
            }
            r[testing + m] += reduce;
        }
    }
}
void FMMFourierTree::aggregate(complex<double>* x, complex<double>* r) {
    int myid = 0;
    int numproc = 1;

    matvecout++;
    //LOWEST-LEVEL AGGREGATION
#pragma omp parallel for
    for (int clusm = myid * numclus[level - 1] / numproc; clusm < (myid + 1) * numclus[level - 1] / numproc; clusm++) {
        int indmulti = clusm * numsamp[level - 1];
        int unk = clusm * box * box;
        for (int k = 0; k < numsamp[level - 1]; k++) {
            complex<double> reduce = 0;
            int indbasis = k * box * box;
            for (int n = 0; n < box * box; n++)
                reduce = reduce + coeff_multi[indbasis + n] * x[unk + n];
            aggmulti[level - 1][indmulti + k] = reduce;
        }
    }
    //HIGHER-LEVEL AGGREGATIONS
    for (int i = level - 2; i > 1; i--) {
#pragma omp parallel for
        for (int clusm = myid * numclus[i] / numproc; clusm < (myid + 1) * numclus[i] / numproc; clusm++) {
            int indm = clusm * numsamp[i];
            for (int km = 0; km < numsamp[i]; km++) {
                complex<double> temp1 = 0;
                for (int cn = 0; cn < 4; cn++) {
                    int indn = (clusm * 4 + cn) * numsamp[i + 1];
                    complex<double> reduce = 0;
                    //INTERPOLATE
                    for (int k = 0; k < ninter; k++)
                        reduce = reduce + interp[i][km * ninter + k] * aggmulti[i + 1][indn + intind[i][km * ninter + k]];
                    //SHIFT
                    temp1 = temp1 + reduce * shiftmul[i][cn * numsamp[i] + km];
                }
                aggmulti[i][indm + km] = temp1;
            }
        }
    }
    int amount = numclus[2] * numsamp[2] / numproc;
    int addres = myid * amount;
    //MPI_Allgathera(&aggmulti[2][addres], amount, MPI_DOUBLE_COMPLEX, agglocal[2], amount, MPI_DOUBLE_COMPLEX, MPI_COMM_MLFMA);
    //memcpy(aggmulti[2], agglocal[2], numclus[2] * numsamp[2] * sizeof(complex<double>));
    //TOP-LEVEL AGGREGATIONS
    for (int i = 1; i > -1; i--) {
#pragma omp parallel for
        for (int clusm = 0; clusm < numclus[i]; clusm++) {
            int indm = clusm * numsamp[i];
            for (int km = 0; km < numsamp[i]; km++) {
                complex<double> temp1 = 0;
                for (int cn = 0; cn < 4; cn++) {
                    int indn = (clusm * 4 + cn) * numsamp[i + 1];
                    complex<double> reduce = 0;
                    //INTERPOLATE
                    for (int k = 0; k < ninter; k++)
                        reduce = reduce + interp[i][km * ninter + k] * aggmulti[i + 1][indn + intind[i][km * ninter + k]];
                    //SHIFT
                    temp1 = temp1 + reduce * shiftmul[i][cn * numsamp[i] + km];
                }
                aggmulti[i][indm + km] = temp1;
            }
        }
    }

#pragma omp parallel for
    for (int m = 0; m < numrx; m++) {
        complex<double> reduce = 0;
        for (int k = 0; k < ninter; k++)
            reduce = reduce + interph[m * ninter + k] * aggmulti[0][intindh[m * ninter + k]];
        r[m] = reduce;
    }

}
void FMMFourierTree::aggregateh(complex<double>* x, complex<double>* r) {
    int myid = 0;
    int numproc = 1;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
#pragma omp parallel for
    for (int m = 0; m < numsamp[0]; m++) {
        complex<double> reduce = 0;
        for (int k = 0; k < 2 * ninter; k++)
            reduce = reduce + anterph[m * 2 * ninter + k] * r[antindh[m * 2 * ninter + k]];
        agglocal[0][m] = reduce * complex<double>(0, -4) * complex<double>(numrx * res * res);
    }
    //TOP-LEVEL DISAGGREGATIONS
    for (int i = 0; i < 2; i++) {
#pragma omp parallel for
        for (int clusn = 0; clusn < numclus[i]; clusn++) {
            int indn = clusn * numsamp[i];
            int mythread = omp_get_thread_num();
            for (int cm = 0; cm < 4; cm++) {
                int indm = (clusn * 4 + cm) * numsamp[i + 1];
                //SHIFT
                for (int kn = 0; kn < numsamp[i]; kn++)
                    temp[mythread][kn] = shiftloc[i][cm * numsamp[i] + kn] * agglocal[i][indn + kn];
                //ANTERPOLATE
                for (int km = 0; km < numsamp[i + 1]; km++) {
                    complex<double> reduce = 0;
                    for (int k = 0; k < 2 * ninter; k++)
                        reduce = reduce + anterp[i + 1][km * 2 * ninter + k] * temp[mythread][antind[i + 1][km * 2 * ninter + k]];
                    agglocal[i + 1][indm + km] = reduce;
                }
            }
        }
    }
    //HIGHER-LEVEL DISAGGREGATIONS
    for (int i = 2; i < level - 1; i++) {
#pragma omp parallel for
        for (int clusn = myid * numclus[i] / numproc; clusn < (myid + 1) * numclus[i] / numproc; clusn++) {
            int indn = clusn * numsamp[i];
            int mythread = omp_get_thread_num();
            for (int cm = 0; cm < 4; cm++) {
                int indm = (clusn * 4 + cm) * numsamp[i + 1];
                //SHIFT
                for (int kn = 0; kn < numsamp[i]; kn++)
                    temp[mythread][kn] = shiftloc[i][cm * numsamp[i] + kn] * agglocal[i][indn + kn];
                //ANTERPOLATE
                for (int km = 0; km < numsamp[i + 1]; km++) {
                    complex<double> reduce = 0;
                    for (int k = 0; k < 2 * ninter; k++)
                        reduce = reduce + anterp[i + 1][km * 2 * ninter + k] * temp[mythread][antind[i + 1][km * 2 * ninter + k]];
                    agglocal[i + 1][indm + km] = reduce;
                }
            }
        }
    }
    //LOWEST-LEVEL DISAGGREGATION
#pragma omp parallel for
    for (int clusm = myid * numclus[level - 1] / numproc; clusm < (myid + 1) * numclus[level - 1] / numproc; clusm++) {
        int unk = clusm * box * box;
        int indlocal = clusm * numsamp[level - 1];
        for (int n = 0; n < box * box; n++) {
            complex<double> reduce = 0;
            int indbasis = n * numsamp[level - 1];
            for (int k = 0; k < numsamp[level - 1]; k++)
                reduce = reduce + basis_local[indbasis + k] * agglocal[level - 1][indlocal + k];
            x[unk + n] = reduce;
        }
    }
}
complex<double> FMMFourierTree::hn(int order, double dist) {
    return complex<double>(jn(order, dist), yn(order, dist));
}
double FMMFourierTree::integ(double x, double y) {
    return -3 * x * y + x * y * log(x * x + y * y) + x * x * atan(y / x) + y * y * atan(x / y);
}
complex<double> FMMFourierTree::integrate(complex<double> post, complex<double> posb) {

    //extern double res;
    complex<double>numer(0, 0);
    complex<double>anal(0, 0);
    //NUMERICAL PART
    double dist = abs(post - posb);
    numer = j0(k0 * dist);
    if (dist < res / 8) {
        numer = numer + complex<double>(0, 0.5772156649015329 * 2 / M_PI + 2 / M_PI * log(0.5));
        //else
        //  numer = numer + complex<double>(0,y0(2*M_PI*dist)-2/M_PI*log(2*M_PI*dist));
          //ANALYTICAL PART
        double xcen = (posb - post).real();
        double ycen = (posb - post).imag();
        double xmin = xcen - res / 2;
        double xmax = xcen + res / 2;
        double ymin = ycen - res / 2;
        double ymax = ycen + res / 2;
        double analt = integ(xmax, ymax) - integ(xmin, ymax) - integ(xmax, ymin) + integ(xmin, ymin);
        anal = complex<double>(0, (analt / 2 + log(k0) * res * res) * 2 / M_PI);
    }
    else
        numer = numer + complex<double>(0, y0(k0 * dist));
    numer = numer * res * res;
    return (numer + anal) * complex<double>(0, 0.25);
}
complex<double> FMMFourierTree::integrate_multi(complex<double> center, complex<double> posb, int order) {
    int numang = numsamp[level - 1];
    double angle = 2 * M_PI * order / numang;
    complex<double> u = complex<double>(cos(angle), sin(angle));
    complex<double>numer(0, 0);
    //NUMERICAL PART
    complex<double> c2s = posb - center;
    numer = exp(complex<double>(0, -k0 * (u.real() * c2s.real() + u.imag() * c2s.imag())));
    return numer * res * res;
}
complex<double> FMMFourierTree::integrate_local(complex<double> center, complex<double> post, int order) {
    int numang = numsamp[level - 1];
    double angle = 2 * M_PI * order / numang;
    complex<double> u = complex<double>(cos(angle), sin(angle));
    complex<double>numer(0, 0);
    //NUMERICAL PART
    complex<double> c2t = post - center;
    numer = exp(complex<double>(0, k0 * (u.real() * c2t.real() + u.imag() * c2t.imag())));
    return numer;
}
void FMMFourierTree::matvec(complex<double>* x, complex<double>* o, complex<double>* b, bool ishermitian) {
    //extern int numunk;
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int amount = numunk / numproc;
    int addres = myid * amount;
    if (ishermitian) {
#pragma omp parallel for
        for (int n = addres; n < addres + amount; n++)
            buffer[n] = conj(x[n]);
        mlfma(buffer, b);
#pragma omp parallel for
        for (int n = addres; n < addres + amount; n++)
            b[n] = x[n] - conj(o[n] * b[n]);
    }
    else {
#pragma omp parallel for
        for (int n = addres; n < addres + amount; n++)
            buffer[n] = o[n] * x[n];
        mlfma(buffer, b);
#pragma omp parallel for
        for (int n = addres; n < addres + amount; n++)
            b[n] = x[n] - b[n];
    }
}
void FMMFourierTree::bicgs(complex<double>* x, complex<double>* o, complex<double>* b, bool ishermitian) {
    int numproc = 1;
    int myid = 0;

    int max_it = iti;
    double res_tol = toli;
    complex<double> alpha;
    complex<double> beta;
    complex<double> omega;
    complex<double> rho;
    complex<double> rho_1;
    int iter = 0;
    //nummatvec = 0;
    double rnrm2;
    double snrm2;
    double bnrm2;
    double error;
    bnrm2 = sqrt(norm2(b));
    if (bnrm2 < pow(10, -20))
        bnrm2 = 1;
    //MATVEC 0 START
    matvec(x, o, r, ishermitian);
    nummatvec++;
    //MATVEC 0 FINISH
    saxpy(b, r, r, complex<double>(-1, 0));
    rnrm2 = sqrt(norm2(r));
    error = rnrm2 / bnrm2;
    //if(myid_world==0)//printf("RES. ERROR: %e ITER: %d\n",error,iter);
    if (error > res_tol) {
        memcpy(&r_tld[myid * numunk / numproc], &r[myid * numunk / numproc], numunk / numproc * sizeof(complex<double>));
        //BEGIN ITERATIONS
        while (iter < max_it) {
            //printf("\t\tBICGS: iter[%d], err=%f\n", iter, error);
            iter++;
            rho = inner(r_tld, r);
            if (abs(rho) < 1e-20) {
                //printf("RHO BREAKDOWN\n");
                break;
            }
            if (iter == 1)
                memcpy(&p[myid * numunk / numproc], &r[myid * numunk / numproc], numunk / numproc * sizeof(complex<double>));
            else {
                beta = (rho / rho_1) * (alpha / omega);
#pragma omp parallel for
                for (int n = myid * numunk / numproc; n < (myid + 1) * numunk / numproc; n++)
                    p[n] = r[n] + beta * (p[n] - omega * v[n]);
            }
            //PRECONDITIONER
            //MATVEC 1 START
            matvec(p, o, v, ishermitian);
            nummatvec++;
            //MATVEC 1 FINISH
            alpha = rho / inner(r_tld, v);
            saxpy(r, v, s, alpha * complex<double>(-1, 0));

            //MATVEC 2 START
            matvec(s, o, t, ishermitian);
            nummatvec++;
            //MATVEC 2 FINISH
            //STABILIZER
            omega = inner(t, s) / norm2(t);
            //UPDATE
#pragma omp parallel for
            for (int n = myid * numunk / numproc; n < (myid + 1) * numunk / numproc; n++) {
                x[n] = x[n] + alpha * p[n] + omega * s[n];
                r[n] = s[n] - omega * t[n];
            }
            rnrm2 = sqrt(norm2(r));
            error = rnrm2 / bnrm2;
            //if(myid_world==0)//printf("RES. ERROR: %e ITER: %d\n",error,iter);
            if (error < res_tol)
                break;
            if (abs(omega) < 1e-20) {
                //printf("OMEGA BREAKDOWN\n");
                break;
            }
            rho_1 = rho;
        }
    }
    /*if (error < res_tol)
        printf("\t\tBICGS: CONVERGED!\n");
    else
        printf("\t\tBICGS: NOT CONVERGED! **************************** iter: %d proc: %d\n", iter, myid);
    */printf("\t\tBICGS: NUMBER OF %d ITERATIONS: %d (%d)\n", ishermitian, iter, nummatvec);
    //printf("\t\tBICGS: RESIDUAL ERROR NORM: %e\n",error);
}
int FMMFourierTree::MoM()
{
    nummatvec = 0;
    int numproc_mlfma = 1;
    int myid_mlfma = 0;

    int amount = numunk / numproc_mlfma;
    int addres = myid_mlfma * amount;
    //INITIAL GUESS (NO OBJECT)
    fill_n(o, numunk, complex<double>(0, 0));
    memcpy(tot, inc, numunk * txproc * sizeof(complex<double>));
#pragma omp parallel for
    for (int m = 0; m < numrx * txproc; m++)
        mesd[m] = -mes[m];

    //GRADIENT
    fill_n(&buff[addres], amount, complex<double>(0, 0));
    for (int i = 0; i < txproc; i++) {
        aggregateh(y, &mesd[i * numrx]);
#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            c[addres + n] = conj(o[addres + n] * k0 * k0) * y[addres + n];
        fill_n(&x[addres], amount, complex<double>(0, 0));
#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            g[addres + n] = o[addres + n] * k0 * k0;
        bicgs(x, g, c, true);
        mlfmah(x, c);
#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            y[addres + n] = y[addres + n] + c[addres + n];
#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            buff[addres + n] = buff[addres + n] + conj(tot[i * numunk + addres + n]) * y[addres + n];
        //REGULARIZE
#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            buff[addres + n] = buff[addres + n] + regular * regular * o[addres + n];
    }
    //MPI_Allreduce(&buff[addres], &od[addres], amount, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_DBIM);
    memcpy(&od[addres], &buff[addres], amount * sizeof(complex<double>));
    memcpy(&ot[addres], &od[addres], amount * sizeof(complex<double>));

    printf("\n**************************** BORN *******************\n");
    for (int iter = 0; iter <= 2; iter++)
    {
        printf("BORN: iter[%d]...", iter);

        //DENOMINATOR
        printf("DENOMINATOR...");
        for (int i = 0; i < txproc; i++) {
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                y[addres + n] = tot[i * numunk + addres + n] * ot[addres + n] * k0 * k0;
            mlfma(y, c);
            fill_n(&x[addres], amount, complex<double>(0, 0));
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                g[addres + n] = o[addres + n] * k0 * k0;
            bicgs(x, g, c, false);

#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                y[addres + n] = y[addres + n] + o[addres + n] * k0 * k0 * x[addres + n];
            aggregate(y, &rhs[i * numrx]);
        }
        complex<double> temp1 = 0;
        complex<double> temp1tot;
        double temp2 = 0;
        double temp2tot;
        for (int i = 0; i < txproc; i++)
            for (int m = 0; m < numrx; m++) {
                temp1 = temp1 + conj(mesd[i * numrx + m]) * rhs[i * numrx + m];
                temp2 = temp2 + norm(rhs[i * numrx + m]);
            }
        //MPI_Allreduce(&temp1, &temp1tot, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_DBIM);
        //MPI_Allreduce(&temp2, &temp2tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_DBIM);
        temp1tot = temp1;
        temp2tot = temp2;
        double alpha = temp1tot.real() / temp2tot;
        printf("alpha: %e...", alpha);

        //TAKE STEP
        printf("TAKE STEP...");
#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            o[addres + n] = o[addres + n] - alpha * ot[addres + n];

        //UPDATE GREEN'S FUNCTION
        for (int i = 0; i < txproc; i++) {
            fill_n(&tot[i * numunk + addres], amount, complex<double>(0, 0));
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                g[addres + n] = o[addres + n] * k0 * k0;
            bicgs(&tot[i * numunk], g, &inc[i * numunk], false);
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                x[addres + n] = tot[i * numunk + addres + n] * o[addres + n] * k0 * k0;
            aggregate(x, &rhs[i * numrx]);
#pragma omp parallel for
            for (int m = 0; m < numrx; m++)
                mesd[i * numrx + m] = rhs[i * numrx + m] - mes[i * numrx + m];
        }
        memcpy(&save[addres], &od[addres], amount * sizeof(complex<double>));

        //GRADIENT
        printf("GRADIENT...");
        fill_n(&buff[addres], amount, complex<double>(0, 0));
        for (int i = 0; i < txproc; i++) {
            aggregateh(y, &mesd[i * numrx]);
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                c[addres + n] = conj(o[addres + n] * k0 * k0) * y[addres + n];
            fill_n(&x[addres], amount, complex<double>(0, 0));
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                g[addres + n] = o[addres + n] * k0 * k0;
            bicgs(x, g, c, true);
            mlfmah(x, c);
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                y[addres + n] = y[addres + n] + c[addres + n];
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                buff[addres + n] = buff[addres + n] + conj(tot[i * numunk + addres + n]) * y[addres + n];
            //REGULARIZE
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                buff[addres + n] = buff[addres + n] + regular * regular * o[addres + n];
        }

        complex<double> norm1 = 0;
        complex<double> norm1tot = 0;
        double norm2 = 0;
        double norm2tot = 0;
        for (int n = 0; n < numunk; n++) {
            norm1 = norm1 + od[n] * conj(od[n] - save[n]);
            norm2 = norm2 + norm(save[n]);
        }
        //MPI_Allreduce(&norm1, &norm1tot, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_MLFMA);
        //MPI_Allreduce(&norm2, &norm2tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_MLFMA);
        norm1tot = norm1;
        norm2tot = norm2;
        complex<double> beta = norm1tot / norm2tot;
        //if (myid == 0)//printf("PR beta: %e %e\n", beta.real(), beta.imag());

#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            ot[addres + n] = od[addres + n] + beta * ot[addres + n];
    }

    return nummatvec;
}
#pragma endregion

#pragma region setup
void FMMFourierTree::setup_mlfma() {

    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    //MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int numproc_mlfma = 1;
    int myid_mlfma = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc_mlfma);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid_mlfma);

    numclus = new int[level];
    clusize = new double[level];
    numsamp = new int[level];
    numterm = new int[level];

    for (int i = 0; i < level; i++)
        numclus[i] = pow(4, i);
    for (int i = 0; i < level; i++)
        clusize[i] = res * box * pow(2, level - i - 1);

    for (int i = 0; i < level; i++) {
        double kr = sqrt(2) * k0 * clusize[i];
        double term = kr + 1.8 * pow(beta, 2.0 / 3) * pow(kr, 1.0 / 3);
        numterm[i] = floor(term);
        numsamp[i] = 4 * numterm[i];
    }

    
    //PREPROCESSING
    pos = new complex<double>[numunk];
    clusc = new complex<double>*[level];
    clusnear = new int* [level];
    clusfar = new int* [level];
    cluspar = new int* [level];
    cluschi = new int* [level];
    for (int i = 0; i < level; i++) {
        clusc[i] = new complex<double>[numclus[i]];
        clusnear[i] = new int[numclus[i] * 9];
        clusfar[i] = new int[numclus[i] * 27];
        cluspar[i] = new int[numclus[i]];
        if (i < level - 1)
            cluschi[i] = new int[numclus[i] * 4];
        else
            cluschi[i] = new int[numclus[i] * box * box];
    }
    unkpar = new int[numunk];
    unkmap = new int[numunk];

    //TREE STRUCTURE
    cluspar[0][0] = -1;
    for (int i = 0; i < level; i++) {
        int lenpar = pow(2, i);
#pragma omp parallel for
        for (int m = 0; m < lenpar; m++)
            for (int n = 0; n < lenpar; n++) {
                int par = m * lenpar + n;
                //HIGHER LEVELS
                if (i < level - 1) {
                    for (int k = 0; k < 2; k++)
                        for (int l = 0; l < 2; l++) {
                            //int chi = m*lenchi*2+n*2+lenchi*k+l;
                            int chi = par * 4 + k * 2 + l;
                            cluschi[i][par * 4 + k * 2 + l] = chi;
                            cluspar[i + 1][chi] = par;
                        }
                }
                //LOWEST LEVEL
                else {
                    for (int k = 0; k < box; k++)
                        for (int l = 0; l < box; l++) {
                            //int chi = m*lenchi*box+n*box+lenchi*k+l;
                            int chi = par * box * box + k * box + l;
                            cluschi[i][par * box * box + k * box + l] = chi;
                            unkpar[chi] = par;
                        }
                }
            }
    }
    //if (myid == 0)//printf("COORDINATES\n");
    //CLUSTER COORDINATES
    clusc[0][0] = 0;
    for (int i = 0; i < level; i++) {
#pragma omp parallel for
        for (int m = 0; m < numclus[i]; m++) {
            if (i < level - 1) {
                complex<double>cor = clusc[i][m] + clusize[i] / 4 * complex<double>(-1, +1);
                for (int k = 0; k < 2; k++)
                    for (int l = 0; l < 2; l++)
                        clusc[i + 1][cluschi[i][m * 4 + k * 2 + l]] = cor + complex<double>(l * clusize[i + 1], -k * clusize[i + 1]);
            }
            //LOWEST LEVEL
            else {
                complex<double>cor = clusc[i][m] + (clusize[i] / 2 - res / 2) * complex<double>(-1, +1);
                for (int k = 0; k < box; k++)
                    for (int l = 0; l < box; l++) {
                        pos[cluschi[i][m * box * box + k * box + l]] = cor + complex<double>(l * res, -k * res);
                    }
            }
        }
    }
    //if (myid == 0)//printf("MAPPING\n");
    //UNKNOWN MAPPING
    complex<double> cor = clusize[0] / 2 * complex<double>(-1, 1);
    int len = box * pow(2, level - 1);
#pragma omp parallel for
    for (int n = 0; n < numunk; n++) {
        int l = floor((pos[n].real() - cor.real()) / res);
        int k = floor((cor.imag() - pos[n].imag()) / res);
        unkmap[k * len + l] = n;
    }
    //if (myid == 0)//printf("NEAR-FAR\n");
    //NEAR-FIELD & FAR-FIELD CLUSTERS
    for (int i = 0; i < level; i++) {
#pragma omp parallel for
        for (int m = 0; m < numclus[i]; m++) {
            for (int n = 0; n < 9; n++)
                clusnear[i][m * 9 + n] = -1;
            for (int n = 0; n < 27; n++)
                clusfar[i][m * 27 + n] = -1;
        }
    }
    clusnear[0][0] = 0;
    for (int i = 1; i < level; i++) {
        double clusdim = clusize[i];
#pragma omp parallel for
        for (int m = 0; m < numclus[i]; m++) {
            int nearid = 0;
            int farid = 0;
            complex<double>posm = clusc[i][m];
            int par = cluspar[i][m];
            for (int ind = 0; ind < 9; ind++) {
                int uncle = clusnear[i - 1][par * 9 + ind];
                if (uncle != -1) {
                    for (int ind2 = 0; ind2 < 4; ind2++) {
                        int n = cluschi[i - 1][uncle * 4 + ind2];
                        if (abs(clusc[i][n] - posm) < clusdim * sqrt(3)) {
                            clusnear[i][m * 9 + nearid] = n;
                            nearid++;
                        }
                        else {
                            clusfar[i][m * 27 + farid] = n;
                            farid++;
                        }
                    }
                }
            }
        }
    }
    //MPI SETUP
    //if (myid == 0)//printf("MPI SETUP\n");
    int* proctemp = new int[numproc_mlfma];
    int** friends = new int* [level + 1];
    int** clustemp = new int* [level + 1];
    for (int i = 0; i < level; i++) {
        friends[i] = new int[numproc_mlfma];
        clustemp[i] = new int[numclus[i]];
        fill_n(friends[i], numproc_mlfma, 0);
        fill_n(clustemp[i], numclus[i], -1);
    }
    friends[level] = new int[numproc_mlfma];
    clustemp[level] = new int[numclus[level - 1]];
    fill_n(friends[level], numproc_mlfma, 0);
    fill_n(clustemp[level], numclus[level - 1], -1);
    //CLUSTEMP & FRIENDS
    for (int i = 0; i < level; i++) {
        for (int clusm = 0; clusm < numclus[i] / numproc_mlfma; clusm++)
            for (int cn = 0; cn < 27; cn++) {
                int clusn = clusfar[i][(numclus[i] / numproc_mlfma * myid_mlfma + clusm) * 27 + cn];
                if (clusn != -1)clustemp[i][clusn] = clusn / (numclus[i] / numproc_mlfma);
            }
        for (int m = 0; m < numclus[i]; m++) {
            int p = clustemp[i][m];
            if (p != -1)friends[i][p]++;
        }
    }
    for (int clusm = 0; clusm < numclus[level - 1] / numproc_mlfma; clusm++)
        for (int cn = 0; cn < 9; cn++) {
            int clusn = clusnear[level - 1][(numclus[level - 1] / numproc_mlfma * myid_mlfma + clusm) * 9 + cn];
            if (clusn != -1)clustemp[level][clusn] = clusn / (numclus[level - 1] / numproc_mlfma);
        }
    for (int m = 0; m < numclus[level - 1]; m++) {
        int p = clustemp[level][m];
        if (p != -1)friends[level][p]++;
    }
    sendmap = new int* [level + 1];
    recvmap = new int* [level + 1];
    procint = new int* [level + 1];
    procmap = new int* [level + 1];
    sendbuff = new complex<double>*[level + 1];
    recvbuff = new complex<double>*[level + 1];
    clusmap = new int* [level + 1];
    clcount = new int[level + 1];
    for (int i = 0; i < level + 1; i++) {
        procint[i] = new int[numproc_mlfma];
        procmap[i] = new int[numproc_mlfma];
        clusmap[i] = new int[numproc_mlfma];
        fill_n(procint[i], numproc_mlfma, -1);
        fill_n(procmap[i], numproc_mlfma, -1);
        fill_n(clusmap[i], numproc_mlfma, 0);
    }
    //PROCTEMP
    {
        int count;
        int me, sib;
        char strtemp[80];
        if (numproc_mlfma > 9)
            sprintf(strtemp, "internodes0%d.dat", numproc_mlfma);
        else
            sprintf(strtemp, "internodes00%d.dat", numproc_mlfma);
        //if (myid == 0)//printf("FILE %s\n", strtemp);
        FILE* mapf = fopen(strtemp, "r");
        fscanf(mapf, "%d", &count);
        count = 0;
        proctemp[count] = myid_mlfma;
        for (int i = 1; i < numproc_mlfma; i++)
            for (int n = 0; n < numproc_mlfma; n++) {
                fscanf(mapf, "%d %d", &me, &sib);
                if (me == myid_mlfma) {
                    count++;
                    proctemp[count] = sib;
                }
            }
        fclose(mapf);
    }
    //PROCMAP
    //if (myid == 0)//printf("PROCMAP\n");
    for (int i = 0; i < level + 1; i++) {
        int numsib = 0;
        int numcls = 0;
        for (int p = 0; p < numproc_mlfma; p++) {
            if (friends[i][proctemp[p]] > 0 && proctemp[p] != myid_mlfma) {
                procmap[i][numsib] = proctemp[p];
                clusmap[i][numsib] = friends[i][proctemp[p]];
                numcls = numcls + friends[i][proctemp[p]];
                numsib++;
            }
        }
        clcount[i] = numcls;
    }
    
    for (int i = 2; i < level + 1; i++) {
        recvmap[i] = new int[clcount[i]];
        sendmap[i] = new int[clcount[i]];
        if (i < level) {
            sendbuff[i] = new complex<double>[clcount[i] * numsamp[i]];
            recvbuff[i] = new complex<double>[clcount[i] * numsamp[i]];
        }
        else {
            sendbuff[i] = new complex<double>[clcount[i] * box * box];
            recvbuff[i] = new complex<double>[clcount[i] * box * box];
        }
    }
    //RECVMAP
    //if (myid == 0)//printf("RECVMAP\n");
    for (int i = 0; i < level + 1; i++) {
        int count = 0;
        for (int p = 0; p < numproc_mlfma; p++) {
            int procm = procmap[i][p];
            if (procm != -1) {
                procint[i][p] = count;
                if (i < level)
                    for (int m = 0; m < numclus[i] / numproc_mlfma; m++) {
                        int clusn = clustemp[i][numclus[i] / numproc_mlfma * procm + m];
                        if (clusn != -1) {
                            recvmap[i][count] = procm * numclus[i] / numproc_mlfma + m;
                            count++;
                        }
                    }
                else
                    for (int m = 0; m < numclus[level - 1] / numproc_mlfma; m++) {
                        int clusn = clustemp[i][numclus[level - 1] / numproc_mlfma * procm + m];
                        if (clusn != -1) {
                            recvmap[i][count] = procm * numclus[level - 1] / numproc_mlfma + m;
                            count++;
                        }
                    }
            }
        }
    }
    
    for (int i = 0; i < level; i++) {
        //delete[] friends[i];
        //delete[] clustemp[i];
    }
    //delete[] proctemp;
    //delete[] friends;
    //delete[] clustemp;
    //SETUP
    complex<double>* hank = new complex<double>[2 * numterm[0] + 1];
    complex<double>* expn = new complex<double>[numterm[0]];
    int* list = new int[numsamp[0]];
    int* neid = new int[numclus[level - 1] * 9];
    near = new complex<double>[9 * box * box * box * box];
    coeff_multi = new complex<double>[numsamp[level - 1] * box * box];
    basis_local = new complex<double>[box * box * numsamp[level - 1]];
    interp = new double* [level];
    anterp = new double* [level];
    intind = new int* [level];
    antind = new int* [level];
    shiftmul = new complex<double>*[level];
    shiftloc = new complex<double>*[level];
    trans = new complex<double>*[level];
    traid = new int* [level];
    for (int i = 0; i < level; i++) {
        interp[i] = new double[numsamp[i] * ninter];
        anterp[i] = new double[numsamp[i] * 2 * ninter];
        intind[i] = new int[numsamp[i] * ninter];
        antind[i] = new int[numsamp[i] * 2 * ninter];
        shiftmul[i] = new complex<double>[4 * numsamp[i]];
        shiftloc[i] = new complex<double>[4 * numsamp[i]];
        trans[i] = new complex<double>[49 * numsamp[i]];
        traid[i] = new int[numclus[i] * 27];
    }
    aggmulti = new complex<double>*[level];
    agglocal = new complex<double>*[level];
    for (int i = 0; i < level; i++) {
        aggmulti[i] = new complex<double>[numclus[i] * numsamp[i]];
        agglocal[i] = new complex<double>[numclus[i] * numsamp[i]];
    }
#pragma omp parallel shared(temp)
    {
        if (omp_get_thread_num() == 0) {
            temp = new complex<double>*[omp_get_num_threads()];
            for (int i = 0; i < omp_get_num_threads(); i++)
                temp[i] = new complex<double>[numsamp[0]];
        }
    }
    //FILL NEARFIELD MATRIX
    //if (myid == 0)//printf("NEARFIELD MATRIX\n");
#pragma omp parallel for
    for (int n = 0; n < box * box * box * box; n++)
        for (int m = 0; m < 9; m++)
            near[n * 9 + m] = 0;
#pragma omp parallel for
    for (int clusm = 0; clusm < numclus[level - 1]; clusm++)
        for (int cn = 0; cn < 9; cn++) {
            int clusn = clusnear[level - 1][clusm * 9 + cn];
            if (clusn != -1) {
                complex<double> m2l = clusc[level - 1][clusm] - clusc[level - 1][clusn];
                int m = round(m2l.imag() / clusize[level - 1]) + 1;
                int n = round(-m2l.real() / clusize[level - 1]) + 1;
                int ind = m * 3 + n;
                neid[clusm * 9 + cn] = ind;
            }
            else
                neid[clusm * 9 + cn] = -1;
        }
    //RESORT
#pragma omp parallel for
    for (int clusm = 0; clusm < numclus[level - 1]; clusm++) {
        int clist[9] = { -1,-1,-1,-1,-1,-1,-1,-1,-1 };
        for (int cn = 0; cn < 9; cn++) {
            int id = neid[clusm * 9 + cn];
            if (id != -1)
                clist[id] = clusnear[level - 1][clusm * 9 + cn];
        }
        for (int cn = 0; cn < 9; cn++)clusnear[level - 1][clusm * 9 + cn] = clist[cn];
        memcpy(&clusnear[level-1][clusm*9],&clist[0],9*sizeof(int));
    }

#pragma omp parallel for
    for (int clusm = 0; clusm < numclus[level - 1]; clusm++)
        for (int cn = 0; cn < 9; cn++) {
            int clusn = clusnear[level - 1][clusm * 9 + cn];
            if (clusn != -1)
                if (near[cn * box * box * box * box] == complex<double>(0))
                    for (int m = 0; m < box * box * box * box; m++) {
                        int k = m / (box * box);
                        int l = m % (box * box);
                        int testing = cluschi[level - 1][clusm * box * box + k];
                        int basis = cluschi[level - 1][clusn * box * box + l];
                        near[cn * box * box * box * box + k * box * box + l] = integrate(pos[testing], pos[basis]);
                    }
        }
    //if (myid == 0)//printf("TRANSLATION OPERATORS\n");
#pragma omp parallel for
    for (int p = 0; p < numterm[0]; p++)
        expn[p] = exp(complex<double>(0, (p + 1) * M_PI / 2));
    //FILL TRANSLATION OPERATORS
    for (int i = 2; i < level; i++) {
        //if (myid == 0)//printf("level %d\n", i + 1);
        double a = clusize[i];
#pragma omp parallel for
        for (int clusm = 0; clusm < numclus[i]; clusm++)
            for (int cn = 0; cn < 27; cn++) {
                int clusn = clusfar[i][clusm * 27 + cn];
                if (clusn != -1) {
                    complex<double> l2m = clusc[i][clusn] - clusc[i][clusm];
                    int n = round(l2m.real() / a) + 3;
                    int m = round(-l2m.imag() / a) + 3;
                    int ind = m * 7 + n;
                    traid[i][clusm * 27 + cn] = ind;
                }
            }
        for (int cm = 0; cm < 7; cm++)
            for (int cn = 0; cn < 7; cn++) {
                complex<double> l2m = -complex<double>(-3 * a + cn * a, 3 * a - cm * a);
                if (abs(l2m) > a * sqrt(3)) {
#pragma omp parallel for
                    for (int p = 0; p < numterm[i] + 1; p++)
                        hank[p] = hn(p, k0 * abs(l2m));
                    int index = cm * 7 * numsamp[i] + cn * numsamp[i];
#pragma omp parallel for
                    for (int k = 0; k < numsamp[i]; k++) {
                        complex<double> reduce = hank[0];
                        for (int p = 1; p < numterm[i] + 1; p++)
                            reduce = reduce + 2 * cos(p * (2 * M_PI * k / numsamp[i] - arg(l2m))) * hank[p] * expn[p - 1];
                        trans[i][index + k] = reduce;
                    }
                }
            }
    }
    //FILL COEFF FOR MULTIPOLE
    //FILL BASIS FOR LOCAL
#pragma omp parallel for
    for (int n = 0; n < box * box; n++) {
        int unk = cluschi[level - 1][n];
        for (int k = 0; k < numsamp[level - 1]; k++) {
            coeff_multi[k * box * box + n] = integrate_multi(clusc[level - 1][0], pos[unk], k);
            basis_local[n * numsamp[level - 1] + k] = integrate_local(clusc[level - 1][0], pos[unk], k) / complex<double>(numsamp[level - 1]) * complex<double>(0, 0.25);
        }
    }
    //FILL SHIFTERS
    //if (myid == 0)//printf("MULTIPOLE & LOCAL SHIFTERS\n");
    for (int i = 0; i < level - 1; i++) {
        int par = 0;
        for (int cn = 0; cn < 4; cn++) {
            int chi = cluschi[i][par * numclus[i] + cn];
            complex<double> rho = clusc[i][par] - clusc[i + 1][chi];
#pragma omp parallel for
            for (int k = 0; k < numsamp[i]; k++) {
                double ang = 2 * M_PI * k / numsamp[i];
                complex<double> kw = k0 * complex<double>(cos(ang), sin(ang));
                double product = kw.real() * rho.real() + kw.imag() * rho.imag();
                shiftmul[i][cn * numsamp[i] + k] = exp(complex<double>(0, product));
                shiftloc[i][cn * numsamp[i] + k] = exp(complex<double>(0, -product));
            }
        }
    }
    //if (myid == 0)//printf("INTERPOLATORS\n");
    //FILL INTERPOLATORS
    for (int i = 0; i < level - 1; i++) {
        double ratio = (double)numsamp[i + 1] / numsamp[i];
#pragma omp parallel for
        for (int m = 0; m < numsamp[i]; m++) {
            int center = 0;
            if (ninter % 2 == 0)
                center = ceil(m * ratio);
            else
                center = round(m * ratio);
            double xm = (double)m / numsamp[i];
            for (int n = 0; n < ninter; n++) {
                int ind = center + n - ninter / 2;
                double mul = 1;
                double xn = (double)ind / numsamp[i + 1];
                for (int k = 0; k < ninter; k++)
                    if (k != n) {
                        double xk = (double)(center + k - ninter / 2) / numsamp[i + 1];
                        mul = mul * (xm - xk) / (xn - xk);
                    }
                interp[i][m * ninter + n] = mul;
                ind = ind % numsamp[i + 1];
                if (ind < 0)
                    ind = ind + numsamp[i + 1];
                intind[i][m * ninter + n] = ind;
            }
        }
    }
    //if (myid == 0)//printf("ANTERPOLATORS\n");
    int* tempi = new int[2 * ninter];
    double* tempd = new double[2 * ninter];
    //FILL ANTERPOLATORS
    for (int i = 1; i < level; i++) {
        for (int n = 0; n < numsamp[0]; n++)
            list[n] = 0;
        for (int m = 0; m < numsamp[i]; m++)
            for (int n = 0; n < 2 * ninter; n++) {
                antind[i][m * 2 * ninter + n] = 0;
                anterp[i][m * 2 * ninter + n] = 0;
            }
        for (int m = 0; m < numsamp[i - 1]; m++)
            for (int n = 0; n < ninter; n++) {
                int ind = intind[i - 1][m * ninter + n];
                antind[i][ind * 2 * ninter + list[ind]] = m;
                anterp[i][ind * 2 * ninter + list[ind]] = interp[i - 1][m * ninter + n] * numsamp[i] / numsamp[i - 1];
                list[ind]++;
            }
        for (int m = 0; m < numsamp[i]; m++)
            for (int n = 1; n < list[m]; n++)
                if (antind[i][m * 2 * ninter + n] != antind[i][m * 2 * ninter + n - 1] + 1) {
                    //if(myid==0)//printf("level %d sample %d index %d(%d) index %d(%d)\n",i,m,n,antind[i][m*2*ninter+n],n-1,antind[i][m*2*ninter+n-1]);
                    memcpy(&tempi[0], &antind[i][m * 2 * ninter + n], (list[m] - n) * sizeof(int));
                    memcpy(&tempd[0], &anterp[i][m * 2 * ninter + n], (list[m] - n) * sizeof(double));
                    memcpy(&tempi[list[m] - n], &antind[i][m * 2 * ninter], n * sizeof(int));
                    memcpy(&tempd[list[m] - n], &anterp[i][m * 2 * ninter], n * sizeof(double));
                    memcpy(&antind[i][m * 2 * ninter], tempi, list[m] * sizeof(int));
                    memcpy(&anterp[i][m * 2 * ninter], tempd, list[m] * sizeof(double));
                    break;
                }
    }
    //delete[] tempi;
    //delete[] tempd;
    //delete[] hank;
    //delete[] list;
    //delete[] expn;
    //delete[] neid;

    setup_interp();
}
void FMMFourierTree::setup_interp() {
    interph = new double[numrx * ninter];
    anterph = new double[numsamp[0] * 2 * ninter];
    intindh = new int[numrx * ninter];
    antindh = new int[numsamp[0] * 2 * ninter];
    int* list = new int[numsamp[0]];
    //TOP-LEVEL INTERPOLATORS
    double ratio = (double)numsamp[0] / numrx;
#pragma omp parallel for
    for (int m = 0; m < numrx; m++) {
        int center = 0;
        if (ninter % 2 == 0)
            center = ceil(m * ratio);
        else
            center = round(m * ratio);
        double xm = (double)m / numrx;
        for (int n = 0; n < ninter; n++) {
            int ind = center + n - ninter / 2;
            double mul = 1;
            double xn = (double)ind / numsamp[0];
            for (int k = 0; k < ninter; k++)
                if (k != n) {
                    double xk = (double)(center + k - ninter / 2) / numsamp[0];
                    mul = mul * (xm - xk) / (xn - xk);
                }
            interph[m * ninter + n] = mul;
            ind = ind % numsamp[0];
            if (ind < 0)
                ind = ind + numsamp[0];
            intindh[m * ninter + n] = ind;
        }
    }
    //TOP-LEVEL ANTERPOLATORS
#pragma omp parallel for
    for (int m = 0; m < numsamp[0]; m++) {
        list[m] = 0;
        for (int n = 0; n < 2 * ninter; n++) {
            antindh[m * 2 * ninter + n] = 0;
            anterph[m * 2 * ninter + n] = 0;
        }
    }
    for (int m = 0; m < numrx; m++)
        for (int n = 0; n < ninter; n++) {
            int ind = intindh[m * ninter + n];
            antindh[ind * 2 * ninter + list[ind]] = m;
            anterph[ind * 2 * ninter + list[ind]] = interph[m * ninter + n] * numsamp[0] / numrx;
            list[ind]++;
        }
    ////delete[] list;
}
void FMMFourierTree::aggregatehsetup_interp() {

    interph = new double[numrx * ninter];
    anterph = new double[numsamp[0] * 2 * ninter];
    intindh = new int[numrx * ninter];
    antindh = new int[numsamp[0] * 2 * ninter];
    int* list = new int[numsamp[0]];
    //TOP-LEVEL INTERPOLATORS
    double ratio = (double)numsamp[0] / numrx;
#pragma omp parallel for
    for (int m = 0; m < numrx; m++) {
        int center = 0;
        if (ninter % 2 == 0)
            center = ceil(m * ratio);
        else
            center = round(m * ratio);
        double xm = (double)m / numrx;
        for (int n = 0; n < ninter; n++) {
            int ind = center + n - ninter / 2;
            double mul = 1;
            double xn = (double)ind / numsamp[0];
            for (int k = 0; k < ninter; k++)
                if (k != n) {
                    double xk = (double)(center + k - ninter / 2) / numsamp[0];
                    mul = mul * (xm - xk) / (xn - xk);
                }
            interph[m * ninter + n] = mul;
            ind = ind % numsamp[0];
            if (ind < 0)
                ind = ind + numsamp[0];
            intindh[m * ninter + n] = ind;
        }
    }
    //TOP-LEVEL ANTERPOLATORS
#pragma omp parallel for
    for (int m = 0; m < numsamp[0]; m++) {
        list[m] = 0;
        for (int n = 0; n < 2 * ninter; n++) {
            antindh[m * 2 * ninter + n] = 0;
            anterph[m * 2 * ninter + n] = 0;
        }
    }
    for (int m = 0; m < numrx; m++)
        for (int n = 0; n < ninter; n++) {
            int ind = intindh[m * ninter + n];
            antindh[ind * 2 * ninter + list[ind]] = m;
            anterph[ind * 2 * ninter + list[ind]] = interph[m * ninter + n] * numsamp[0] / numrx;
            list[ind]++;
        }
    //delete[] list;
}
void FMMFourierTree::setup_bicgs() {
    ////ALLOCATIONS
    p = new complex<double>[numunk];
    v = new complex<double>[numunk];
    s = new complex<double>[numunk];
    t = new complex<double>[numunk];
    r = new complex<double>[numunk];
    r_tld = new complex<double>[numunk];
    buffer = new complex<double>[numunk];

    //p = (complex<double>*)malloc(numunk * sizeof(complex<double>));
    //v = (complex<double>*)malloc(numunk * sizeof(complex<double>));
    //s = (complex<double>*)malloc(numunk * sizeof(complex<double>));
    //t = (complex<double>*)malloc(numunk * sizeof(complex<double>));
    //r_tld = (complex<double>*)malloc(numunk * sizeof(complex<double>));
    //buffer = (complex<double>*)malloc(numunk * sizeof(complex<double>));
}
void FMMFourierTree::setup_born() {
    int numproc = 1;
    int myid = 0;

    for (int i = 0; i < txproc; i++)
#pragma omp parallel for
        for (int n = 0; n < numunk; n++)
            inc[i * numunk + n] = exp(complex<double>(0, k0 * (tx[mytx + i].real() * pos[n].real() + tx[mytx + i].imag() * pos[n].imag())));

    //generating random input instead of loading phantom.txt
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int n = 0; n < numunk; n++) {
        o[unkmap[n]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
    }


    for (int i = 0; i < txproc; i++) {
        fill_n(x, numunk, complex<double>(0, 0));
#pragma omp parallel for
        for (int n = 0; n < numunk; n++)
            g[n] = o[n] * k0 * k0;
        bicgs(x, g, &inc[i * numunk], false);
#pragma omp parallel for
        for (int n = 0; n < numunk; n++)
            x[n] = x[n] * g[n];
        aggregate(x, &mes[i * numrx]);
    }

    objnorm = 0;
    memcpy(omes, o, numunk * sizeof(complex<double>));
    for (int n = 0; n < numunk; n++)
        objnorm = objnorm + norm(omes[n]);
    double err = 0;
    for (int i = 0; i < txproc; i++) {
        for (int m = 0; m < numrx; m++)
            err = err + norm(mes[i * numrx + m]);
    }
}
#pragma endregion


void FMMFourierTree::generateRandomMeasuredField()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    generatedField = vector<complex<double>>(numunk);
    for (int n = 0; n < numunk; n++) {
        o[unkmap[n]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
        generatedField[unkmap[n]] = o[unkmap[n]];
    }
}
void FMMFourierTree::bicgs_mvp(complex<double>* x, complex<double>* o, complex<double>* b, bool ishermitian) {
    int numproc = 1;
    int myid = 0;

    int max_it = iti;
    double res_tol = toli;
    complex<double> alpha;
    complex<double> beta;
    complex<double> omega;
    complex<double> rho;
    complex<double> rho_1;
    int iter = 0;
    //nummatvec = 0;
    double rnrm2;
    double snrm2;
    double bnrm2;
    double error;
    bnrm2 = sqrt(norm2(b));
    if (bnrm2 < pow(10, -20))
        bnrm2 = 1;
    //MATVEC 0 START
    matvec_mvp(x, o, r);
    nummatvec++;
    //MATVEC 0 FINISH
    saxpy(b, r, r, complex<double>(-1, 0));
    rnrm2 = sqrt(norm2(r));
    error = rnrm2 / bnrm2;
    //if(myid_world==0)//printf("RES. ERROR: %e ITER: %d\n",error,iter);
    if (error > res_tol) {
        memcpy(&r_tld[myid * numunk / numproc], &r[myid * numunk / numproc], numunk / numproc * sizeof(complex<double>));
        //BEGIN ITERATIONS
        while (iter < max_it) {
            printf("\t\tBICGS: iter[%d], err=%f\n", iter, error);
            iter++;
            rho = inner(r_tld, r);
            if (abs(rho) < 1e-20) {
                printf("RHO BREAKDOWN\n");
                break;
            }
            if (iter == 1)
                memcpy(&p[myid * numunk / numproc], &r[myid * numunk / numproc], numunk / numproc * sizeof(complex<double>));
            else {
                beta = (rho / rho_1) * (alpha / omega);
#pragma omp parallel for
                for (int n = myid * numunk / numproc; n < (myid + 1) * numunk / numproc; n++)
                    p[n] = r[n] + beta * (p[n] - omega * v[n]);
            }
            //PRECONDITIONER
            //MATVEC 1 START
            matvec_mvp(p, o, v);
            nummatvec++;
            //MATVEC 1 FINISH
            alpha = rho / inner(r_tld, v);
            saxpy(r, v, s, alpha * complex<double>(-1, 0));

            //MATVEC 2 START
            matvec_mvp(s, o, t);
            nummatvec++;
            //MATVEC 2 FINISH
            //STABILIZER
            omega = inner(t, s) / norm2(t);
            //UPDATE
#pragma omp parallel for
            for (int n = myid * numunk / numproc; n < (myid + 1) * numunk / numproc; n++) {
                x[n] = x[n] + alpha * p[n] + omega * s[n];
                r[n] = s[n] - omega * t[n];
            }
            rnrm2 = sqrt(norm2(r));
            error = rnrm2 / bnrm2;
            //if(myid_world==0)//printf("RES. ERROR: %e ITER: %d\n",error,iter);
            if (error < res_tol)
                break;
            if (abs(omega) < 1e-20) {
                printf("OMEGA BREAKDOWN\n");
                break;
            }
            rho_1 = rho;
        }
    }
    if (error < res_tol)
        printf("\t\tBICGS: CONVERGED!\n");
    else
        printf("\t\tBICGS: NOT CONVERGED! **************************** iter: %d proc: %d\n", iter, myid);
    printf("\t\tBICGS: NUMBER OF %d ITERATIONS: %d (%d)\n", ishermitian, iter, nummatvec);
    printf("\t\tBICGS: RESIDUAL ERROR NORM: %e\n", error);
}
void FMMFourierTree::getEmpedanceMatrix()
{
    int n = pow(4, level);
    //complex<double>* __z = new complex<double>[n * n];

    for (int clusm = 0; clusm < numclus[level - 1]; clusm++)
    {
        for (int clusn = 0; clusn < numclus[level - 1]; clusn++)
        {
            int testing = cluschi[level - 1][clusm];
            int basis = cluschi[level - 1][clusn];
            __z[clusm * n + clusn] = integrate(pos[testing], pos[basis]);
        }
    }
}
void FMMFourierTree::matvec_mvp(complex<double>* x, complex<double>* o, complex<double>* b) 
{
    //extern int numunk;
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int amount = numunk / numproc;
    int addres = myid * amount;
    for (int n = addres; n < addres + amount; n++)
        buffer[n] = o[n] * x[n];
    for (int i = 0; i < numunk; i++)
    {
        for (int j = 0; j < numunk; j++)
        {
            complex<double> sum = 0;
            for (int k = 0; k < numunk; k++)
            {
                 sum += buffer[i * numunk + j] * __z[k * numunk + j];
            }
            b[i * numunk + j] = sum;
        }
    }
    for (int n = addres; n < addres + amount; n++)
        b[n] = x[n] - b[n];
}

