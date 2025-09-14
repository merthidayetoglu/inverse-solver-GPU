#include "FMMFourierTreeGPU.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern int verbosity;

FMMFourierTreeGPU::FMMFourierTreeGPU(int N, int PointsPerBox)
{
    //level = L;
    box = sqrt(PointsPerBox);
    level = log(N / (box * box)) / log(4);

    int numproc = 1;
    int myid = 0;

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
    
    //FREQUENCIES
    scales = new double[numfreq];
    for (int i = 0; i < numfreq; i++) {
        scale = maxfreq - (maxfreq - minfreq) / (numfreq - 1) * i;
        if (numfreq == 1)scale = maxfreq;
        scales[i] = scale;
    }
    k0 = 2 * M_PI * scales[myfreq];
    dim = res * box * pow(2, level - 1);
    numunk = pow(dim / res, 2);
    if(verbosity >= 1) printf("FmmTree : N=%d, L=%d : ", numunk, level);

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
    if (verbosity >= 1) printf("SETUP MLFMA...");
    setup_mlfma();
    setup_interp();
    if (verbosity >= 1) printf("SETUP GPU...");
    setup_gpu();
    /*if (verbosity >= 1) printf("SETUP BICGS...");
    setup_bicgs();*/
    if (verbosity >= 1) printf("ALLOCATIONS...");

    inc = new complex<double>[txproc * numunk];
    tot = new complex<double>[txproc * numunk];
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
}

#pragma region solve GPU 
int FMMFourierTreeGPU::MoM(complex<double>* measuredField)
{
    nummatvec = 0;
    vector<complex<double>>vi;
    ot_i.shrink_to_fit();
    ot_i.clear();

    if (measuredField == NULL)
        generateRandomMeasuredField();
    else
        memcpy(o, measuredField, numunk * sizeof(complex<double>));

    totalCopyTime = 0.0;
    totalDuplicationTime = 0.0;
    totalComputationTime = 0.0;
    totalNearFieldTime = 0.0;
    totalFarFieldTime = 0.0;
    totalMlfmaTime = 0.0;

    //if(verbosity>=1) 
        printf("SETUP BICGS...");
    setup_bicgs();
    //if (verbosity >= 1) 
        printf("SETUP BORN...");
    setup_born();
    //if (verbosity >= 1) 
        printf("SETUP FINISHED...");

    int numproc_mlfma = 1;
    int myid_mlfma = 0;
    int amount = numunk / numproc_mlfma;
    int addres = myid_mlfma * amount;
    //INITIAL GUESS (NO OBJECT)
    if (verbosity >= 1) printf("INITIAL GUESS(NO OBJECT)\n");
    fill_n(o, numunk, complex<double>(0, 0));
    memcpy(tot, inc, numunk * txproc * sizeof(complex<double>));
#pragma omp parallel for
    for (int m = 0; m < numrx * txproc; m++)
        mesd[m] = -mes[m];

    //GRADIENT
    if (verbosity >= 1) printf("GRADIENT\n");
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
        bicgs(x, g, c, true, verbosity >= 2);
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
    //memcpy(&ot[addres], &od[addres], amount * sizeof(complex<double>));
    memcpy(&ot[addres], &buff[addres], amount * sizeof(complex<double>));
    memcpy(&od[addres], &buff[addres], amount * sizeof(complex<double>));


    if (verbosity >= 1) printf("\n\n**************************** BORN *******************\n");
    for (int iter = 0; iter <= bornIteratoins; iter++) {
        printf("Born: iter[%d]...", iter);
        //DENOMINATOR
        if (verbosity >= 1) printf("DENOMINATOR...");
        for (int i = 0; i < txproc; i++) {
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                y[addres + n] = tot[i * numunk + addres + n] * ot[addres + n] * k0 * k0;
            mlfma(y, c);
            fill_n(&x[addres], amount, complex<double>(0, 0));
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                g[addres + n] = o[addres + n] * k0 * k0;
            bicgs(x, g, c, false, verbosity >= 2);

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
        if (verbosity >= 1) printf("alpha=%3e...", alpha);
        //TAKE STEP
        if (verbosity >= 1) printf("TAKE STEP...");
#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            o[addres + n] = o[addres + n] - alpha * ot[addres + n];

        //UPDATE GREEN'S FUNCTION
        if (verbosity >= 1) printf("UPDATE GREEN'S FUNCTION...");
        for (int i = 0; i < txproc; i++) {
            fill_n(&tot[i * numunk + addres], amount, complex<double>(0, 0));
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                g[addres + n] = o[addres + n] * k0 * k0;
            bicgs(&tot[i * numunk], g, &inc[i * numunk], false, verbosity >= 2);
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
        if (verbosity >= 1) printf("GRADIENT...");
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
            bicgs(x, g, c, true, verbosity >= 2);
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
        if (verbosity >= 1) printf("PR beta=<%3e,%3e>\n", beta.real(), beta.imag());

#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            ot[addres + n] = od[addres + n] + beta * ot[addres + n];


        //or image reconstrungction visualization
        for (int i = 0; i < numunk; i++) vi.push_back(ot[i]);
        ot_i.push_back(vi);
        vi.shrink_to_fit();
        vi.clear();
    }

    return nummatvec;
}
void FMMFourierTreeGPU::mlfma(complex<double>* x, complex<double>* r) {
    int numproc = 1;
    int myid = 0;
    int memory;
    int addres;
    float miliseconds=0, t1=0,t2=0,t3=0;

#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    //TRANSFER DATA IN
    memory = numunk / numproc * sizeof(cuDoubleComplex);
    addres = myid * numunk / numproc;
    cudaMemcpy(&x_d[addres], &x[addres], memory, cudaMemcpyHostToDevice);
    cudaMemcpy(&r_d[addres], &r[addres], memory, cudaMemcpyHostToDevice);

#if defined(_DEBUG)
#else
    //cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    t1 = (miliseconds / 1000);
    cudaEventRecord(start);
#endif

    mlfma_gpu(x_d, r_d);
    cudaDeviceSynchronize();

#if defined(_DEBUG)
#else
    //cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    t2 = (miliseconds / 1000);
    cudaEventRecord(start);
#endif
    cudaMemcpy(&r[addres], &r_d[addres], memory, cudaMemcpyDeviceToHost);
#if defined(_DEBUG)
#else
    //cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    t3 = (miliseconds / 1000);
#endif

    totalCopyTime += (t1 + t3);
    //totalMlfmaTime += (t2 + totalCopyTime);
    totalMlfmaTime += t2;
}
void FMMFourierTreeGPU::mlfmah(complex<double>* x, complex<double>* r) {
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int memory;
    int addres;
#pragma omp parallel for
    for (int n = myid * numunk / numproc; n < (myid + 1) * numunk / numproc; n++)
        x[n] = conj(x[n]);
    //TRANSFER DATA IN
    memory = numunk / numproc * sizeof(cuDoubleComplex);
    addres = myid * numunk / numproc;
    cudaMemcpy(&x_d[addres], &x[addres], memory, cudaMemcpyHostToDevice);
    mlfma_gpu(x_d, r_d);
    cudaMemcpy(&r[addres], &r_d[addres], memory, cudaMemcpyDeviceToHost);
#pragma omp parallel for
    for (int n = myid * numunk / numproc; n < (myid + 1) * numunk / numproc; n++) {
        r[n] = conj(r[n]);
        x[n] = conj(x[n]);
    }
}
void FMMFourierTreeGPU::mlfma_gpu(cuDoubleComplex* x_d, cuDoubleComplex* r_d) 
{
    float miliseconds;
#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    mlfma_gpu_farField(x_d, r_d);
    cudaDeviceSynchronize();
#if defined(_DEBUG)
#else
    //cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    totalFarFieldTime += (miliseconds / 1000);

    cudaEventRecord(start);
#endif   
    mlfma_gpu_nearField(x_d, r_d);
    cudaDeviceSynchronize();
#if defined(_DEBUG)
#else
    //cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    totalNearFieldTime += (miliseconds / 1000);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif  
}
void FMMFourierTreeGPU::mlfma_gpu_farField(cuDoubleComplex* x_d, cuDoubleComplex* r_d)
{
    int numproc = 1;
    int myid = 0;

    float miliseconds;
    int memory;
    int addres;
    int addrss;
    int tile;
    int tilex;
    int tiley;
    /*cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    matvecin++;
    cudaEventRecord(start);*/

    //LOWEST-LEVEL AGGREGATION
    tile = 16;
    memory = 2 * tile * tile * sizeof(cuDoubleComplex);
    addres = myid * numunk / numproc;
    addrss = myid * numclus[level - 1] / numproc * numsamp[level - 1];
    dim3 P2Mgrid(numsamp[level - 1] / tile, (numclus[level - 1] / numproc) / tile, 1);
    if (numsamp[level - 1] % tile)P2Mgrid.x++;
    if ((numclus[level - 1] / numproc) % tile)P2Mgrid.y++;
    dim3 P2Mblock(tile, tile, 1);
    P2Mkernel(&aggmulti_d[level - 1][addrss], &x_d[addres], coeff_d, numclus[level - 1], box * box, numsamp[level - 1], P2Mgrid, P2Mblock, memory, kerrstr);
    //P2Mkernel << <P2Mgrid, P2Mblock, memory, kerrstr >> > (&aggmulti_d[level - 1][addrss], &x_d[addres], coeff_d, numclus[level - 1], box * box, numsamp[level - 1]);
    //HIGHER-LEVEL AGGREGATION
    for (int i = level - 2; i > 1; i--) {
        if (numsamp[i] < 1024) {
            memory = numsamp[i + 1] * 4 * sizeof(cuDoubleComplex);
            addres = myid * numclus[i] / numproc * numsamp[i];
            addrss = myid * numclus[i + 1] / numproc * numsamp[i + 1];
            //dim3 M2Mgrid(sqrt(numclus[i]/numproc),sqrt(numclus[i]/numproc),1);
            dim3 M2Mgrid(numclus[i] / numproc, 1, 1);
            dim3 M2Mblock(numsamp[i], 1, 1);
            M2Mkernel(&aggmulti_d[i][addres], &aggmulti_d[i + 1][addrss], numsamp[i + 1], interp_d[i], intind_d[i], ninter, shiftmul_d[i], M2Mgrid, M2Mblock, memory, kerrstr);
            //M2Mkernel << <M2Mgrid, M2Mblock, memory, kerrstr >> > (&aggmulti_d[i][addres], &aggmulti_d[i + 1][addrss], numsamp[i + 1], interp_d[i], intind_d[i], ninter, shiftmul_d[i]);
        }
        else {
            tilex = 16;
            tiley = 16;
            memory = (ninter + 4) * tilex * sizeof(cuDoubleComplex) + tilex * sizeof(int);
            addres = myid * numclus[i] / numproc * numsamp[i];
            addrss = myid * numclus[i + 1] / numproc * numsamp[i + 1];
            dim3 M2Mgrid(numsamp[i] / tilex, (numclus[i] / numproc) / tiley, 1);
            if (numsamp[i] % tilex)M2Mgrid.x++;
            if ((numclus[i] / numproc) % tiley)M2Mgrid.y++;
            dim3 M2Mblock(tilex, tiley, 1);
            //M2Mkernel_output << <M2Mgrid, M2Mblock, memory, kerrstr >> > (&aggmulti_d[i][addres], &aggmulti_d[i + 1][addrss], numsamp[i], numsamp[i + 1], numclus[i] / numproc, interp_d[i], intind_d[i], ninter, shiftmul_d[i]);
            M2Mkernel_output(&aggmulti_d[i][addres], &aggmulti_d[i + 1][addrss], numsamp[i], numsamp[i + 1], numclus[i] / numproc, interp_d[i], intind_d[i], ninter, shiftmul_d[i], M2Mgrid, M2Mblock, memory, kerrstr);
        }
    }

    //TRANSLATION
    for (int i = level - 1; i > 1; i--) {
        tilex = 8;
        tiley = 16;
        memory = 2 * tiley * 27 * sizeof(int);
        addres = myid * numclus[i] / numproc * numsamp[i];
        addrss = myid * numclus[i] / numproc * 27;
        dim3 M2Lgrid(numsamp[i] / tilex, (numclus[i] / numproc) / tiley, 1);
        if (numsamp[i] % tilex)M2Lgrid.x++;
        if ((numclus[i] / numproc) % tiley)M2Lgrid.y++;
        dim3 M2Lblock(tilex, tiley, 1);
        //M2Lkernel_output << <M2Lgrid, M2Lblock, memory >> > (&agglocal_d[i][addres], aggmulti_d[i], numsamp[i], numclus[i] / numproc, &clusfar_d[i][addrss], &traid_d[i][addrss], trans_d[i]);
        M2Lkernel_output(&agglocal_d[i][addres], aggmulti_d[i], numsamp[i], numclus[i] / numproc, &clusfar_d[i][addrss], &traid_d[i][addrss], trans_d[i], M2Lgrid, M2Lblock, memory);
    }
    //HIGHER-LEVEL DISAGGREGATION
    for (int i = 2; i < level - 1; i++) {
        if (numsamp[i] > 512) {
            tilex = 4;
            tiley = 16;
            memory = 2 * ninter * tilex * sizeof(cuDoubleComplex) + tilex * sizeof(int);
            addres = myid * numclus[i] / numproc * numsamp[i];
            addrss = myid * numclus[i + 1] / numproc * numsamp[i + 1];
            dim3 L2Lgrid(numsamp[i + 1] / tilex, numclus[i] / tiley, 1);
            if (numsamp[i + 1] % tilex)L2Lgrid.x++;
            if (numclus[i] % tiley)L2Lgrid.y++;
            dim3 L2Lblock(tilex, tiley, 1);
            //L2Lkernel_output << <L2Lgrid, L2Lblock, memory >> > (&agglocal_d[i][addres], &agglocal_d[i + 1][addrss], numsamp[i], numsamp[i + 1], numclus[i] / numproc, anterp_d[i + 1], antind_d[i + 1], 2 * ninter, shiftloc_d[i]);
            L2Lkernel_output(&agglocal_d[i][addres], &agglocal_d[i + 1][addrss], numsamp[i], numsamp[i + 1], numclus[i] / numproc, anterp_d[i + 1], antind_d[i + 1], 2 * ninter, shiftloc_d[i], L2Lgrid, L2Lblock, memory);
        }
        else {
            memory = 4 * numsamp[i] * sizeof(cuDoubleComplex);
            addres = myid * numclus[i] / numproc * numsamp[i];
            addrss = myid * numclus[i + 1] / numproc * numsamp[i + 1];
            //dim3 L2Lgrid(sqrt(numclus[i]/numproc),sqrt(numclus[i]/numproc),1);
            dim3 L2Lgrid(numclus[i] / numproc, 1, 1);
            dim3 L2Lblock(numsamp[i], 1, 1);
            //L2Lkernel << <L2Lgrid, L2Lblock, memory >> > (&agglocal_d[i][addres], &agglocal_d[i + 1][addrss], numsamp[i + 1], anterp_d[i + 1], antind_d[i + 1], 2 * ninter, shiftloc_d[i]);
            L2Lkernel(&agglocal_d[i][addres], &agglocal_d[i + 1][addrss], numsamp[i + 1], anterp_d[i + 1], antind_d[i + 1], 2 * ninter, shiftloc_d[i], L2Lgrid, L2Lblock, memory);
        }
    }
    //LOWEST-LEVEL DISAGGREGATION
    tile = 16;
    memory = 2 * tile * tile * sizeof(cuDoubleComplex);
    addres = myid * numunk / numproc;
    addrss = myid * numclus[level - 1] / numproc * numsamp[level - 1];
    dim3 L2Pgrid((box * box) / tile, (numclus[level - 1] / numproc) / tile, 1);
    if ((box * box) % tile)L2Pgrid.x++;
    if ((numclus[level - 1] / numproc) % tile)L2Pgrid.y++;
    dim3 L2Pblock(tile, tile, 1);
    //L2Pkernel << <L2Pgrid, L2Pblock, memory >> > (&r_d[addres], &agglocal_d[level - 1][addrss], basis_d, numclus[level - 1], numsamp[level - 1], box * box);
    L2Pkernel(&r_d[addres], &agglocal_d[level - 1][addrss], basis_d, numclus[level - 1], numsamp[level - 1], box * box, L2Pgrid, L2Pblock, memory);

    //cudaDeviceSynchronize();
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&miliseconds, start, stop);
    ////innert = innert + miliseconds / 1000;
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
}
void FMMFourierTreeGPU::mlfma_gpu_nearField(cuDoubleComplex* x_d, cuDoubleComplex* r_d)
{
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);

    float miliseconds;
    int memory;
    int addres;
    int addrss;
    int tile;
    int tilex;
    int tiley;

    memory = numunk * sizeof(cuDoubleComplex);
#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
//    cudaMemcpy(x_d, x, memory, cudaMemcpyHostToDevice);
//#if defined(_DEBUG)
//#else
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&miliseconds, start, stop);
//    totalCopyTime += (miliseconds / 1000);
//
//    cudaEventRecord(start);
//#endif
  
//NEARFIELD MULTIPLICATION
    int chunkSize = (int)max(1.0, 8.0 / (double)box * 4);
    memory = chunkSize * 9 * sizeof(int) + 2 * chunkSize * box * box * sizeof(cuDoubleComplex);
    addres = myid * numunk / numproc;
    addrss = myid * numclus[level - 1] / numproc * 9;
    dim3 P2Pgrid(numclus[level - 1] / numproc / chunkSize, 1, 1);
    dim3 P2Pblock(chunkSize * box * box, 1, 1);
    //P2Pkernel << <P2Pgrid, P2Pblock, memory, kerrstr >> > (&r_d[addres], x_d, &clusnear_d[addrss], near_d);
    P2Pkernel(&r_d[addres], x_d, &clusnear_d[addrss], near_d, chunkSize, P2Pgrid, P2Pblock, memory, kerrstr);
    //P2Pkernel_myModification(&r_d[addres], x_d, &clusnear_d[addrss], near_d, P2Pgrid, P2Pblock, memory, kerrstr);
    cudaError_t et = cudaDeviceSynchronize();
    if (et != cudaSuccess)
    {
        printf("\nError executing p2P kernel : %s\n", cudaGetErrorString(et));
        return;
    }
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    totalComputationTime += (miliseconds / 1000);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}
void FMMFourierTreeGPU::aggregate(complex<double>* x, complex<double>* r) {
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);

    float miliseconds;
    int memory;
    int addres;
    int addrss;
    int tile;
    int tilex;
    int tiley;
#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif
    matvecout++;

    //TRANSFER DATA IN
    memory = numunk / numproc * sizeof(cuDoubleComplex);
    addres = myid * numunk / numproc;
    cudaMemcpy(&x_d[addres], &x[addres], memory, cudaMemcpyHostToDevice);

    //LOWEST-LEVEL AGGREGATION
    tile = 16;
    memory = 2 * tile * tile * sizeof(cuDoubleComplex);
    addres = myid * numunk / numproc;
    addrss = myid * numclus[level - 1] / numproc * numsamp[level - 1];
    dim3 P2Mgrid(numsamp[level - 1] / tile, (numclus[level - 1] / numproc) / tile, 1);
    if (numsamp[level - 1] % tile)P2Mgrid.x++;
    if ((numclus[level - 1] / numproc) % tile)P2Mgrid.y++;
    dim3 P2Mblock(tile, tile, 1);
    //P2Mkernel << <P2Mgrid, P2Mblock, memory, kerrstr >> > (&aggmulti_d[level - 1][addrss], &x_d[addres], coeff_d, numclus[level - 1], box * box, numsamp[level - 1]);
    P2Mkernel(&aggmulti_d[level - 1][addrss], &x_d[addres], coeff_d, numclus[level - 1], box * box, numsamp[level - 1], P2Mgrid, P2Mblock, memory, kerrstr);
    //HIGHER-LEVEL AGGREGATION
    for (int i = level - 2; i > 1; i--) {
        if (numsamp[i] < 1024) {
            memory = numsamp[i + 1] * 4 * sizeof(cuDoubleComplex);
            addres = myid * numclus[i] / numproc * numsamp[i];
            addrss = myid * numclus[i + 1] / numproc * numsamp[i + 1];
            //dim3 M2Mgrid(sqrt(numclus[i]/numproc),sqrt(numclus[i]/numproc),1);
            dim3 M2Mgrid(numclus[i] / numproc, 1, 1);
            dim3 M2Mblock(numsamp[i], 1, 1);
            //M2Mkernel << <M2Mgrid, M2Mblock, memory, kerrstr >> > (&aggmulti_d[i][addres], &aggmulti_d[i + 1][addrss], numsamp[i + 1], interp_d[i], intind_d[i], ninter, shiftmul_d[i]);
            M2Mkernel(&aggmulti_d[i][addres], &aggmulti_d[i + 1][addrss], numsamp[i + 1], interp_d[i], intind_d[i], ninter, shiftmul_d[i], M2Mgrid, M2Mblock, memory, kerrstr);
        }
        else {
            tilex = 16;
            tiley = 16;
            memory = (ninter + 4) * tilex * sizeof(cuDoubleComplex) + tilex * sizeof(int);
            addres = myid * numclus[i] / numproc * numsamp[i];
            addrss = myid * numclus[i + 1] / numproc * numsamp[i + 1];
            dim3 M2Mgrid(numsamp[i] / tilex, (numclus[i] / numproc) / tiley, 1);
            if (numsamp[i] % tilex)M2Mgrid.x++;
            if ((numclus[i] / numproc) % tiley)M2Mgrid.y++;
            dim3 M2Mblock(tilex, tiley, 1);
            //M2Mkernel_output << <M2Mgrid, M2Mblock, memory, kerrstr >> > (&aggmulti_d[i][addres], &aggmulti_d[i + 1][addrss], numsamp[i], numsamp[i + 1], numclus[i] / numproc, interp_d[i], intind_d[i], ninter, shiftmul_d[i]);
            M2Mkernel_output(&aggmulti_d[i][addres], &aggmulti_d[i + 1][addrss], numsamp[i], numsamp[i + 1], numclus[i] / numproc, interp_d[i], intind_d[i], ninter, shiftmul_d[i], M2Mgrid, M2Mblock, memory, kerrstr);
        }
    }
    //TRANSFER DATA OUT
    int amount = numclus[2] * numsamp[2] / numproc;
    addres = myid * amount;
    memory = amount * sizeof(cuDoubleComplex);
    cudaMemcpy(&aggmulti[2][addres], &aggmulti_d[2][addres], memory, cudaMemcpyDeviceToHost);
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
    //extern double* interph;
    //extern int* intindh;
    //extern int numrx;
    //TOP-LEVEL INTERPOLATION
#pragma omp parallel for
    for (int m = 0; m < numrx; m++) {
        complex<double> reduce = 0;
        for (int k = 0; k < ninter; k++)
            reduce = reduce + interph[m * ninter + k] * aggmulti[0][intindh[m * ninter + k]];
        r[m] = reduce;
    }
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    outert = outert + miliseconds / 1000;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}
void FMMFourierTreeGPU::aggregateh(complex<double>* x, complex<double>* r) {
    int myid = 0;
    int numproc = 1;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);

    float miliseconds;
    int memory;
    int addres;
    int addrss;
    int tile;
    int tilex;
    int tiley;
#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif
    matvecout++;
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
    //TRANSFER DATA IN
    int amount = numclus[2] * numsamp[2] / numproc;
    addres = myid * amount;
    memory = amount * sizeof(cuDoubleComplex);
    cudaMemcpy(&agglocal_d[2][addres], &agglocal[2][addres], memory, cudaMemcpyHostToDevice);
    //HIGHER-LEVEL DISAGGREGATION
    for (int i = 2; i < level - 1; i++) {
        if (numsamp[i] > 512) {
            tilex = 4;
            tiley = 16;
            memory = 2 * ninter * tilex * sizeof(cuDoubleComplex) + tilex * sizeof(int);
            addres = myid * numclus[i] / numproc * numsamp[i];
            addrss = myid * numclus[i + 1] / numproc * numsamp[i + 1];
            dim3 L2Lgrid(numsamp[i + 1] / tilex, numclus[i] / tiley, 1);
            if (numsamp[i + 1] % tilex)L2Lgrid.x++;
            if (numclus[i] % tiley)L2Lgrid.y++;
            dim3 L2Lblock(tilex, tiley, 1);
            //L2Lkernelh_output << <L2Lgrid, L2Lblock, memory >> > (&agglocal_d[i][addres], &agglocal_d[i + 1][addrss], numsamp[i], numsamp[i + 1], numclus[i] / numproc, anterp_d[i + 1], antind_d[i + 1], 2 * ninter, shiftloc_d[i]);
            L2Lkernelh_output(&agglocal_d[i][addres], &agglocal_d[i + 1][addrss], numsamp[i], numsamp[i + 1], numclus[i] / numproc, anterp_d[i + 1], antind_d[i + 1], 2 * ninter, shiftloc_d[i], L2Lgrid, L2Lblock, memory);
        }
        else {
            memory = 4 * numsamp[i] * sizeof(cuDoubleComplex);
            addres = myid * numclus[i] / numproc * numsamp[i];
            addrss = myid * numclus[i + 1] / numproc * numsamp[i + 1];
            //dim3 L2Lgrid(sqrt(numclus[i]/numproc),sqrt(numclus[i]/numproc),1);
            dim3 L2Lgrid(numclus[i] / numproc, 1, 1);
            dim3 L2Lblock(numsamp[i], 1, 1);
            //L2Lkernelh << <L2Lgrid, L2Lblock, memory >> > (&agglocal_d[i][addres], &agglocal_d[i + 1][addrss], numsamp[i + 1], anterp_d[i + 1], antind_d[i + 1], 2 * ninter, shiftloc_d[i]);
            L2Lkernelh(&agglocal_d[i][addres], &agglocal_d[i + 1][addrss], numsamp[i + 1], anterp_d[i + 1], antind_d[i + 1], 2 * ninter, shiftloc_d[i], L2Lgrid, L2Lblock, memory);
        }
    }
    //LOWEST-LEVEL DISAGGREGATION
    tile = 16;
    memory = 2 * tile * tile * sizeof(cuDoubleComplex);
    addres = myid * numunk / numproc;
    addrss = myid * numclus[level - 1] / numproc * numsamp[level - 1];
    dim3 L2Pgrid((box * box) / tile, (numclus[level - 1] / numproc) / tile, 1);
    if ((box * box) % tile)L2Pgrid.x++;
    if ((numclus[level - 1] / numproc) % tile)L2Pgrid.y++;
    dim3 L2Pblock(tile, tile, 1);
    //L2Pkernelh << <L2Pgrid, L2Pblock, memory >> > (&r_d[addres], &agglocal_d[level - 1][addrss], basis_d, numclus[level - 1], numsamp[level - 1], box * box);
    L2Pkernelh(&r_d[addres], &agglocal_d[level - 1][addrss], basis_d, numclus[level - 1], numsamp[level - 1], box * box, L2Pgrid, L2Pblock, memory);
    //TRANSFER DATA OUT
    memory = numunk / numproc * sizeof(cuDoubleComplex);
    cudaMemcpy(&x[addres], &r_d[addres], memory, cudaMemcpyDeviceToHost);
   
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    outert = outert + miliseconds / 1000;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}
complex<double> FMMFourierTreeGPU::hn(int order, double dist){
  return complex<double>(jn(order,dist),yn(order,dist));
}
double FMMFourierTreeGPU::integ(double x, double y){
  return -3*x*y+x*y*log(x*x+y*y)+x*x*atan(y/x)+y*y*atan(x/y);
}
complex<double> FMMFourierTreeGPU::integrate(complex<double> post, complex<double> posb){

  //extern double res;
  complex<double>numer(0,0);
  complex<double>anal(0,0);
  //NUMERICAL PART
  double dist = abs(post-posb);
  numer = j0(k0*dist);
  if(dist < res/8){
    numer = numer + complex<double>(0,0.5772156649015329*2/M_PI+2/M_PI*log(0.5));
  //else
  //  numer = numer + complex<double>(0,y0(2*M_PI*dist)-2/M_PI*log(2*M_PI*dist));
    //ANALYTICAL PART
    double xcen=(posb-post).real();
    double ycen=(posb-post).imag();
    double xmin=xcen-res/2;
    double xmax=xcen+res/2;
    double ymin=ycen-res/2;
    double ymax=ycen+res/2;
    double analt=integ(xmax,ymax)-integ(xmin,ymax)-integ(xmax,ymin)+integ(xmin,ymin);
    anal=complex<double>(0,(analt/2+log(k0)*res*res)*2/M_PI);
  }
  else
    numer =  numer + complex<double>(0,y0(k0*dist));
  numer = numer*res*res;
  return (numer+anal)*complex<double>(0,0.25);
}
complex<double> FMMFourierTreeGPU::integrate_multi(complex<double> center, complex<double> posb, int order){
  //extern double res;
  //extern int *numsamp;
  //extern int level;
  int numang = numsamp[level-1];
  double angle = 2*M_PI*order/numang;
  complex<double> u = complex<double>(cos(angle),sin(angle));
  complex<double>numer(0,0);
  //NUMERICAL PART
  complex<double> c2s = posb-center;
  numer = exp(complex<double>(0,-k0*(u.real()*c2s.real()+u.imag()*c2s.imag())));
  return numer*res*res;
}
complex<double> FMMFourierTreeGPU::integrate_local(complex<double> center, complex<double> post, int order){
  //extern double res;
  //extern int *numsamp;
  //extern int level;
  int numang = numsamp[level-1];
  double angle = 2*M_PI*order/numang;
  complex<double> u = complex<double>(cos(angle),sin(angle));
  complex<double>numer(0,0);
  //NUMERICAL PART
  complex<double> c2t = post-center;
  numer = exp(complex<double>(0,k0*(u.real()*c2t.real()+u.imag()*c2t.imag())));
  return numer;
}
void FMMFourierTreeGPU::farfield(complex<double>* x) {

    //extern int level;
    //extern int box;

    //extern int ninter;
    //extern int* numsamp;
    //extern int* numclus;

    //extern complex<double>* coeff_multi;
    //extern complex<double>** aggmulti;
    //extern double** interp;
    //extern int** intind;
    //extern complex<double>** shiftmul;

    //LOWEST-LEVEL AGGREGATION
#pragma omp parallel for
    for (int clusm = 0; clusm < numclus[level - 1]; clusm++) {
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
    for (int i = level - 2; i > -1; i--) {
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
        //printf("level %d\n", i);
    }
    //INTERPOLATION
    int numsampf = 36000;
    double* interpf = new double[numsampf * ninter];
    int* intindf = new int[numsampf * ninter];
    complex<double>* pattern = new complex<double>[numsampf];
    double ratio = (double)numsamp[0] / numsampf;
#pragma omp parallel for
    for (int m = 0; m < numsampf; m++) {
        int center = 0;
        if (ninter % 2 == 0)
            center = ceil(m * ratio);
        else
            center = round(m * ratio);
        double xm = (double)m / numsampf;
        for (int n = 0; n < ninter; n++) {
            int ind = center + n - ninter / 2;
            double mul = 1;
            double xn = (double)ind / numsamp[0];
            for (int k = 0; k < ninter; k++)
                if (k != n) {
                    double xk = (double)(center + k - ninter / 2) / numsamp[0];
                    mul = mul * (xm - xk) / (xn - xk);
                }
            interpf[m * ninter + n] = mul;
            ind = ind % numsamp[0];
            if (ind < 0)
                ind = ind + numsamp[0];
            intindf[m * ninter + n] = ind;
        }
    }
    for (int m = 0; m < numsampf; m++) {
        complex<double> reduce = 0;
        for (int k = 0; k < ninter; k++)
            reduce = reduce + interpf[m * ninter + k] * aggmulti[0][intindf[m * ninter + k]];
        pattern[m] = reduce;
    }
    FILE* test = fopen("far.bin", "wb");
    fwrite(pattern, sizeof(complex<double>), numsampf, test);
    fclose(test);
    //delete[] interpf;
    //delete[] intindf;
}
double FMMFourierTreeGPU::norm2(cuDoubleComplex* a) {
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int addres = myid * numunk / numproc;
    int blocksize = 128;
    int memory = 2 * blocksize * sizeof(double);
    int numblocks = numunk / numproc / (2 * blocksize);
    //norm2partc << <numblocks, blocksize, memory >> > (&dbuff[addres], &a[addres]);
    norm2partc(&dbuff[addres], &a[addres], numblocks, blocksize, memory);
    while (numblocks / (2 * blocksize) > 0) {
        numblocks = numblocks / (2 * blocksize);
        //norm2partd << <numblocks, blocksize, memory >> > (&dbuff[addres]);
        norm2partd(&dbuff[addres], numblocks, blocksize, memory);
    }
    cudaMemcpy(&dbuff_h[addres], &dbuff[addres], numblocks * sizeof(double), cudaMemcpyDeviceToHost);
    double reduce = 0;
    for (int m = 0; m < numblocks; m++)
        reduce = reduce + dbuff_h[addres + m];
    //double rettot;
    //MPI_Allreduce(&reduce, &rettot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_MLFMA);
    //return rettot;
    return reduce;
}
complex<double> FMMFourierTreeGPU::inner(cuDoubleComplex* a, cuDoubleComplex* b) {
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int addres = myid * numunk / numproc;
    int blocksize = 128;
    int memory = 2 * blocksize * sizeof(cuDoubleComplex);
    int numblocks = numunk / numproc / (2 * blocksize);
    //innerpartc << <numblocks, blocksize, memory >> > (&cbuff[addres], &a[addres], &b[addres]);
    innerpartc(&cbuff[addres], &a[addres], &b[addres], numblocks, blocksize, memory);
    while (numblocks / (2 * blocksize) > 0) {
        numblocks = numblocks / (2 * blocksize);
        //innerpartd << <numblocks, blocksize, memory >> > (&cbuff[addres]);
        innerpartd(&cbuff[addres], numblocks, blocksize, memory);
    }
    cudaMemcpy(&cbuff_h[addres], &cbuff[addres], numblocks * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    complex<double> reduce = 0;
    for (int m = 0; m < numblocks; m++)
        reduce = reduce + cbuff_h[addres + m];
    //complex<double> rettot;
    //MPI_Allreduce(&reduce, &rettot, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_MLFMA);
    //return rettot;
    return reduce;
}
void FMMFourierTreeGPU::matvec(cuDoubleComplex* x, cuDoubleComplex* o, cuDoubleComplex* r, bool ishermitian) {
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int addres = myid * numunk / numproc;
    if (ishermitian) {
        //prep << <numunk / numproc / 256, 256 >> > (&cbuff[addres], &x[addres]);
        prep(&cbuff[addres], &x[addres], dim3(numunk / numproc / 256), dim3(256));
        mlfma_gpu(cbuff, r);
        //post << <numunk / numproc / 256, 256 >> > (&r[addres], &x[addres], &o[addres]);
        post(&r[addres], &x[addres], &o[addres], dim3(numunk / numproc / 256), dim3(256));
    }
    else {
        //prep << <numunk / numproc / 256, 256 >> > (&cbuff[addres], &x[addres], &o[addres]);
        prep(&cbuff[addres], &x[addres], &o[addres], dim3(numunk / numproc / 256), dim3(256));
        mlfma_gpu(cbuff, r);
        //post << <numunk / numproc / 256, 256 >> > (&r[addres], &x[addres]);
        post(&r[addres], &x[addres], dim3(numunk / numproc / 256), dim3(256));
    }
}
void FMMFourierTreeGPU::direct(complex<double>* x, complex<double>* r)
{
    //to measure the progression
    //int ls = numclus[level - 1] * box * box;
    //int totc = pow(ls, 2);
    //int totc100 = totc / 100;
    int totc100 = (int)floor((double)numclus[level - 1] / 100.0);
    int prog = 0;

    //int c = 0;
    for (int clusm = 0; clusm < numclus[level - 1]; clusm++)
    {
        if (clusm % totc100 == 0) printf("\rProgress: [%d%%]", ++prog);

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

            //c += ls;
            //if (c >= totc100)
            //{
            //    prog++;
            //    printf("\rProgress: [%d%%]", prog);
            //    c = 0;
            //}
        }
    }
}

#pragma endregion

#pragma region setup gpu
void FMMFourierTreeGPU::setup_mlfma() {

    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    //MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int numproc_mlfma;
    int myid_mlfma;
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
    numproc_mlfma = 1;
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
    myid_mlfma = 0;
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
        memset(aggmulti[i], 0, numclus[i] * numsamp[i] * sizeof(complex<double>));
        memset(agglocal[i], 0, numclus[i] * numsamp[i] * sizeof(complex<double>));
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
    //RESORT neaighbours based on type of radius's direction
#pragma omp parallel for
    for (int clusm = 0; clusm < numclus[level - 1]; clusm++) {
        int clist[9] = { -1,-1,-1,-1,-1,-1,-1,-1,-1 };
        for (int cn = 0; cn < 9; cn++) {
            int id = neid[clusm * 9 + cn];
            if (id != -1)
                clist[id] = clusnear[level - 1][clusm * 9 + cn];
        }
        for (int cn = 0; cn < 9; cn++)clusnear[level - 1][clusm * 9 + cn] = clist[cn];
        memcpy(&clusnear[level - 1][clusm * 9], &clist[0], 9 * sizeof(int));
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

}
FMMFourierTreeGPU::~FMMFourierTreeGPU()
{
    /*//free(near);
    //free(coeff_multi);
    //free(basis_local);
    //free(interp);
    //free(anterp);
    //free(intind);
    //free(antind);
    //free(shiftmul);
    //free(shiftloc);
    //free(trans);
    //free(traid);
    //free(aggmulti);
    //free(agglocal);

    //free(interph);
    //free(anterph);
    //free(intindh);
    //free(antindh);
    //free(aggmulti_d);
    //free(agglocal_d);
    //free(interp_d);
    //free(anterp_d);
    //free(intind_d);
    //free(antind_d);
    //free(shiftmul_d);
    //free(shiftloc_d);
    //free(trans_d);
    //free(traid_d);
    //free(clusfar_d);
    //free(sendmap_d);
    //free(recvmap_d);
    //free(clcount_d);
    //free(sendbuff_d);
    //free(recvbuff_d);
    //free(sendbuff_h);
    //free(recvbuff_h);*/

    cudaFree(x_h);
    cudaFree(r_h);
    cudaFree(x_d);
    cudaFree(r_d);
    cudaFree(coeff_d);
    cudaFree(basis_d);
    cudaFree(near_d);
    cudaFree(clusnear_d);
    
    cudaFree(p);
    cudaFree(v);
    cudaFree(s);
    cudaFree(t);
    cudaFree(r_tld);
    cudaFree(cbuff);
    cudaFree(dbuff);
    cudaFree(cbuff_h);
    cudaFree(dbuff_h);
    cudaFree(o_d);
    cudaFree(b_d);

    cudaStreamDestroy(commstr);
}
void FMMFourierTreeGPU::setup_interp() {

    //extern int numrx;
    //extern int ninter;
    //extern int* numsamp;
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
void FMMFourierTreeGPU::setup_gpu() {
    //printf("setting device...");
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if(verbosity >= 1) printf("Device Number: %d\n", i);
        if (verbosity >= 1) printf("  Device name: %s\n", prop.name);
        //printf("  Memory Clock Rate (KHz): %d\n",
        //    prop.memoryClockRate);
        //printf("  Memory Bus Width (bits): %d\n",
        //    prop.memoryBusWidth);
        //printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
        //    2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        //
        ////cout<<"  Max Blocks Grid Size: " << prop.maxGridSize << endl;
        //printf("  Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
        //printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        //printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        //printf("  Shared memory per SM: %d\n", prop.sharedMemPerMultiprocessor);
        //printf("  Shared memory per Block: %d\n", prop.sharedMemPerBlock);
        //cout<<"  total global mem: " << prop.totalGlobalMem << endl;
        //printf("  total constant mem: %d\n", prop.totalConstMem);
    }
    cudaSetDevice(0);
    //cudaDeviceReset();

    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    //printf("creating streams...\n");
    cudaError_t e =  cudaStreamCreate(&commstr);
    if (e != cudaSuccess)
    {
        printf("GPU ERROR!! maybe cuda device wes not set properly : %s", cudaGetErrorString(e));
    }
    cudaStreamCreate(&kerrstr);
    complex<double>* coeff_h = new complex<double>[box * box * numsamp[level - 1]];
    complex<double>* basis_h = new complex<double>[box * box * numsamp[level - 1]];
#pragma omp parallel for
    for (int m = 0; m < numsamp[level - 1]; m++)
        for (int n = 0; n < box * box; n++) {
            coeff_h[n * numsamp[level - 1] + m] = coeff_multi[m * box * box + n];
            basis_h[m * box * box + n] = basis_local[n * numsamp[level - 1] + m];
        }
    complex<double>* near_h = new complex<double>[9 * box * box * box * box];
    for (int k = 0; k < 9; k++) {
        int start = k * box * box * box * box;
#pragma omp parallel for
        for (int m = 0; m < box * box; m++)
            for (int n = 0; n < box * box; n++)
                near_h[start + m * box * box + n] = near[start + n * box * box + m];
    }
    int** intind_h = new int* [level];
    double** interp_h = new double* [level];
    for (int i = 0; i < level - 1; i++) {
        intind_h[i] = new int[numsamp[i]];
        interp_h[i] = new double[numsamp[i] * ninter];
#pragma omp parallel for
        for (int l = 0; l < numsamp[i]; l++) {
            intind_h[i][l] = intind[i][l * ninter];
            for (int n = 0; n < ninter; n++)
                interp_h[i][n * numsamp[i] + l] = interp[i][l * ninter + n];
        }
    }
    int** antind_h = new int* [level];
    double** anterp_h = new double* [level];
    for (int i = 1; i < level; i++) {
        antind_h[i] = new int[numsamp[i]];
        anterp_h[i] = new double[numsamp[i] * 2 * ninter];
#pragma omp parallel for
        for (int l = 0; l < numsamp[i]; l++) {
            antind_h[i][l] = antind[i][l * 2 * ninter];
            for (int n = 0; n < 2 * ninter; n++)
                anterp_h[i][n * numsamp[i] + l] = anterp[i][l * 2 * ninter + n];
        }
    }
    gpumem = 0;
    //printf("allocating mem...\n");
    cudaMallocHost((void**)&x_h, numunk * sizeof(complex<double>));
    cudaMallocHost((void**)&r_h, numunk * sizeof(complex<double>));
    cudaMalloc((void**)&x_d, numunk * sizeof(complex<double>));
    cudaMalloc((void**)&r_d, numunk * sizeof(complex<double>));
    gpumem = gpumem + (double)sizeof(complex<double>) * 2 / 1024 / 1024;
    cudaMalloc((void**)&coeff_d, numsamp[level - 1] * box * box * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&basis_d, numsamp[level - 1] * box * box * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&near_d, 9 * box * box * box * box * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&clusnear_d, numclus[level - 1] * 9 * sizeof(int));
    gpumem = gpumem + (double)sizeof(cuDoubleComplex) * 2 * numsamp[level - 1] * box * box / 1024 / 1024;
    gpumem = gpumem + (double)sizeof(cuDoubleComplex) * 9 * box * box * box * box / 1024 / 1024;
    gpumem = gpumem + (double)sizeof(int) * 9 * numclus[level - 1] / 1024 / 1024;
    aggmulti_d = new cuDoubleComplex * [level];
    agglocal_d = new cuDoubleComplex * [level];
    interp_d = new double* [level];
    anterp_d = new double* [level];
    intind_d = new int* [level];
    antind_d = new int* [level];
    shiftmul_d = new cuDoubleComplex * [level];
    shiftloc_d = new cuDoubleComplex * [level];
    trans_d = new cuDoubleComplex * [level];
    traid_d = new int* [level];
    clusfar_d = new int* [level];
    gpumem = gpumem + (double)sizeof(cuDoubleComplex*) * 5 * level / 1024 / 1024;
    gpumem = gpumem + (double)sizeof(double*) * 2 * level / 1024 / 1024;
    gpumem = gpumem + (double)sizeof(int*) * 4 * level / 1024 / 1024;

    sendmap_d = new int* [level + 1];
    recvmap_d = new int* [level + 1];
    clcount_d = new int[level + 1];
    sendbuff_d = new cuDoubleComplex * [level + 1];
    recvbuff_d = new cuDoubleComplex * [level + 1];
    sendbuff_h = new complex<double>*[level + 1];
    recvbuff_h = new complex<double>*[level + 1];
    gpumem = gpumem + (double)sizeof(int*) * 2 * (level + 1) / 1024 / 1024;
    gpumem = gpumem + (double)sizeof(cuDoubleComplex*) * 2 * (level + 1) / 1024 / 1024;
    gpumem = gpumem + (double)sizeof(int) * (level + 1) / 1024 / 1024;
    for (int i = 2; i < level + 1; i++) {
        cudaMalloc((void**)&sendmap_d[i], clcount[i] * sizeof(int));
        cudaMalloc((void**)&recvmap_d[i], clcount[i] * sizeof(int));
        gpumem = gpumem + (double)sizeof(int) * 2 * clcount[i] / 1024 / 1024;
        if (i < level) {
            cudaMalloc((void**)&sendbuff_d[i], clcount[i] * numsamp[i] * sizeof(cuDoubleComplex));
            cudaMalloc((void**)&recvbuff_d[i], clcount[i] * numsamp[i] * sizeof(cuDoubleComplex));
            cudaMallocHost((void**)&sendbuff_h[i], clcount[i] * numsamp[i] * sizeof(complex<double>));
            cudaMallocHost((void**)&recvbuff_h[i], clcount[i] * numsamp[i] * sizeof(complex<double>));
            gpumem = gpumem + (double)sizeof(cuDoubleComplex) * 2 * clcount[i] * numsamp[i] / 1024 / 1024;
        }
        else {
            cudaMalloc((void**)&sendbuff_d[i], clcount[i] * box * box * sizeof(cuDoubleComplex));
            cudaMalloc((void**)&recvbuff_d[i], clcount[i] * box * box * sizeof(cuDoubleComplex));
            cudaMallocHost((void**)&sendbuff_h[i], clcount[i] * box * box * sizeof(complex<double>));
            cudaMallocHost((void**)&recvbuff_h[i], clcount[i] * box * box * sizeof(complex<double>));
            gpumem = gpumem + (double)sizeof(cuDoubleComplex) * 2 * clcount[i] * box * box / 1024 / 1024;
        }
    }
    for (int i = 0; i < level; i++) {
        cudaMalloc((void**)&aggmulti_d[i], numclus[i] * numsamp[i] * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&agglocal_d[i], numclus[i] * numsamp[i] * sizeof(cuDoubleComplex));
        gpumem = gpumem + (double)sizeof(cuDoubleComplex) * 2 * numclus[i] * numsamp[i] / 1024 / 1024;
        if (i > 1) {
            cudaMalloc((void**)&trans_d[i], 49 * numsamp[i] * sizeof(cuDoubleComplex));
            cudaMalloc((void**)&traid_d[i], 27 * numclus[i] * sizeof(int));
            cudaMalloc((void**)&clusfar_d[i], 27 * numclus[i] * sizeof(int));
            gpumem = gpumem + (double)sizeof(cuDoubleComplex) * 49 * numsamp[i] / 1024 / 1024;
            gpumem = gpumem + (double)sizeof(int) * 2 * 27 * numclus[i] / 1024 / 1024;
        }
    }
    for (int i = 0; i < level - 1; i++) {
        cudaMalloc((void**)&interp_d[i], numsamp[i] * ninter * sizeof(double));
        cudaMalloc((void**)&anterp_d[i + 1], 2 * numsamp[i + 1] * ninter * sizeof(double));
        cudaMalloc((void**)&intind_d[i], numsamp[i] * sizeof(int));
        cudaMalloc((void**)&antind_d[i + 1], numsamp[i + 1] * sizeof(int));
        cudaMalloc((void**)&shiftmul_d[i], 4 * numsamp[i] * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&shiftloc_d[i], 4 * numsamp[i] * sizeof(cuDoubleComplex));
        gpumem = gpumem + (double)sizeof(double) * numsamp[i] * ninter / 1024 / 1024;
        gpumem = gpumem + (double)sizeof(double) * 2 * numsamp[i + 1] * ninter / 1024 / 1024;
        gpumem = gpumem + (double)sizeof(int) * numsamp[i] / 1024 / 1024;
        gpumem = gpumem + (double)sizeof(int) * 2 * numsamp[i + 1] / 1024 / 1024;
        gpumem = gpumem + (double)sizeof(cuDoubleComplex) * 2 * 4 * numsamp[i] / 1024 / 1024;
    }
    //if (myid == 0)//printf("GPU MEMORY: %f MB\n", gpumem);
    int memory;
    memory = (level + 1) * sizeof(int);
    //printf("copy data...\n");
    cudaMemcpy(clcount_d, clcount, memory * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 2; i < level + 1; i++) {
        memory = clcount[i] * sizeof(int);
        cudaMemcpy(sendmap_d[i], sendmap[i], memory, cudaMemcpyHostToDevice);
        cudaMemcpy(recvmap_d[i], recvmap[i], memory, cudaMemcpyHostToDevice);
    }
    memory = numsamp[level - 1] * box * box * sizeof(cuDoubleComplex);
    cudaMemcpy(coeff_d, coeff_h, memory, cudaMemcpyHostToDevice);
    cudaMemcpy(basis_d, basis_h, memory, cudaMemcpyHostToDevice);
    memory = numclus[level - 1] * 9 * sizeof(int);
    cudaMemcpy(clusnear_d, clusnear[level - 1], memory, cudaMemcpyHostToDevice);
    memory = 9 * box * box * box * box * sizeof(cuDoubleComplex);
    cudaMemcpy(near_d, near_h, memory, cudaMemcpyHostToDevice);
    for (int i = 2; i < level - 1; i++) {
        memory = numsamp[i] * ninter * sizeof(double);
        cudaMemcpy(interp_d[i], interp_h[i], memory, cudaMemcpyHostToDevice);
        memory = 2 * numsamp[i + 1] * ninter * sizeof(double);
        cudaMemcpy(anterp_d[i + 1], anterp_h[i + 1], memory, cudaMemcpyHostToDevice);
        memory = numsamp[i] * sizeof(int);
        cudaMemcpy(intind_d[i], intind_h[i], memory, cudaMemcpyHostToDevice);
        memory = numsamp[i + 1] * sizeof(int);
        cudaMemcpy(antind_d[i + 1], antind_h[i + 1], memory, cudaMemcpyHostToDevice);
        memory = 4 * numsamp[i] * sizeof(cuDoubleComplex);
        cudaMemcpy(shiftmul_d[i], shiftmul[i], memory, cudaMemcpyHostToDevice);
        cudaMemcpy(shiftloc_d[i], shiftloc[i], memory, cudaMemcpyHostToDevice);
    }
    for (int i = 2; i < level; i++) {
        memory = 49 * numsamp[i] * sizeof(cuDoubleComplex);
        cudaMemcpy(trans_d[i], trans[i], memory, cudaMemcpyHostToDevice);
        memory = 27 * numclus[i] * sizeof(int);
        cudaMemcpy(traid_d[i], traid[i], memory, cudaMemcpyHostToDevice);
        cudaMemcpy(clusfar_d[i], clusfar[i], memory, cudaMemcpyHostToDevice);
    }
    //delete[] coeff_h;
    //delete[] basis_h;
    //delete[] near_h;
    for (int i = 0; i < level - 1; i++) {
        //delete[] intind_h[i];
        //delete[] antind_h[i + 1];
    }
    //delete[] intind_h;
    //delete[] antind_h;
}
void FMMFourierTreeGPU::setup_bicgs() {

    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    //MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    cudaMalloc((void**)&p, numunk * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&v, numunk * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&s, numunk * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&t, numunk * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&r_tld, numunk * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&cbuff, numunk * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&dbuff, numunk * sizeof(double));
    cudaMallocHost((void**)&cbuff_h, numunk * sizeof(cuDoubleComplex));
    cudaMallocHost((void**)&dbuff_h, numunk * sizeof(double));
    cudaMalloc((void**)&o_d, numunk * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&b_d, numunk * sizeof(cuDoubleComplex));

    //extern double gpumem;
    double memtemp = 0;
    memtemp = memtemp + (double)sizeof(cuDoubleComplex) * 10 * numunk / 1024.0 / 1024.0;
    memtemp = memtemp + (double)sizeof(double) * numunk / 1024.0 / 1024.0;
    //if (myid == 0)//printf("GPU BICGSTAB SOLVER MEM: %f MB\n", memtemp);
    //if (myid == 0)//printf("GPU TOTAL MEM: %f MB\n", gpumem + memtemp);
}
void FMMFourierTreeGPU::bicgs(complex<double>* x, complex<double>* o, complex<double>* b, bool ishermitian, bool verbos) {

    if(verbos) printf("\nBICGSTAB:\n");
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int numproc_world;
    int myid_world;

    int addres = myid * numunk / numproc;
    cudaMemcpy(&o_d[addres], &o[addres], numunk / numproc * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(&b_d[addres], &b[addres], numunk / numproc * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(&x_d[addres], &x[addres], numunk / numproc * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    int max_it = iti;
    double res_tol = toli;
    complex<double> alpha;
    complex<double> beta;
    complex<double> omega;
    complex<double> rho;
    complex<double> rho_1;

    int iter = 0;
    double rnrm2;
    double snrm2;
    double bnrm2;
    double error;

    bnrm2 = sqrt(norm2(b_d));
    if (bnrm2 < 1e-20)
        bnrm2 = 1;
    //MATVEC 0 START
    matvec(x_d, o_d, r_d, ishermitian);
    nummatvec++;
    //MATVEC 0 FINISH
    //saxp << <numunk / numproc / 256, 256 >> > (&b_d[addres], &r_d[addres], &r_d[addres], make_cuDoubleComplex(-1, 0));
    saxp(&b_d[addres], &r_d[addres], &r_d[addres], make_cuDoubleComplex(-1, 0), dim3(numunk / numproc / 256), dim3(256));
    rnrm2 = sqrt(norm2(r_d));
    error = rnrm2 / bnrm2;
    //if(myid_world==0)//printf("RES. ERROR: %e ITER: %d\n",error,iter);
    if (error > res_tol) {
        //saxp << <numunk / numproc / 256, 256 >> > (&r_d[addres], &r_d[addres], &r_tld[addres], make_cuDoubleComplex(0, 0));
        saxp(&r_d[addres], &r_d[addres], &r_tld[addres], make_cuDoubleComplex(0, 0), dim3(numunk / numproc / 256), dim3(256));
        //BEGIN ITERATIONS
        while (iter < max_it) {
            if (verbos) printf("\t\tBICGSTAB: iter[%d], error=%f\n", iter, error);
            iter++;
            rho = inner(r_tld, r_d);
            if (abs(rho) < 1e-200) {
                if (verbos) printf("\t\tBICGSTAB: RHO BREAKDOWN\n");
                break;
            }
            if (iter == 1)
                //saxp << <numunk / numproc / 256, 256 >> > (&r_d[addres], &r_d[addres], &p[addres], make_cuDoubleComplex(0, 0));
                saxp(&r_d[addres], &r_d[addres], &p[addres], make_cuDoubleComplex(0, 0), dim3(numunk / numproc / 256), dim3(256));
            else {
                beta = (rho / rho_1) * (alpha / omega);
                double omegar = omega.real();
                double omegai = omega.imag();
                //saxp << <numunk / numproc / 256, 256 >> > (&p[addres], &v[addres], &cbuff[addres], make_cuDoubleComplex(-1 * omegar, -1 * omegai));
                saxp(&p[addres], &v[addres], &cbuff[addres], make_cuDoubleComplex(-1 * omegar, -1 * omegai), dim3(numunk / numproc / 256), dim3(256));
                double betar = beta.real();
                double betai = beta.imag();
                //saxp << <numunk / numproc / 256, 256 >> > (&r_d[addres], &cbuff[addres], &p[addres], make_cuDoubleComplex(betar, betai));
                saxp(&r_d[addres], &cbuff[addres], &p[addres], make_cuDoubleComplex(betar, betai), dim3(numunk / numproc / 256), dim3(256));
            }
            //PRECONDITIONER
            //MATVEC 1 START
            matvec(p, o_d, v, ishermitian);
            nummatvec++;
            //MATVEC 1 FINISH
            alpha = rho / inner(r_tld, v);
            double alphar = alpha.real();
            double alphai = alpha.imag();
            //saxp << <numunk / numproc / 256, 256 >> > (&r_d[addres], &v[addres], &s[addres], make_cuDoubleComplex(-1 * alphar, -1 * alphai));
            saxp(&r_d[addres], &v[addres], &s[addres], make_cuDoubleComplex(-1 * alphar, -1 * alphai), dim3(numunk / numproc / 256), dim3(256));
            //MATVEC 2 START
            matvec(s, o_d, t, ishermitian);
            nummatvec++;
            //MATVEC 2 FINISH
            //STABILIZER
            omega = inner(t, s) / norm2(t);
            complex<double> lan1 = inner(t, s);
            double lan2 = norm2(t);
            if (abs(lan1) < 1e-200 && lan2 < 1e-200)omega = 1.0;//PREVENT NAN
            //UPDATE
            //saxp << <numunk / numproc / 256, 256 >> > (&x_d[addres], &p[addres], &x_d[addres], make_cuDoubleComplex(alphar, alphai));
            saxp(&x_d[addres], &p[addres], &x_d[addres], make_cuDoubleComplex(alphar, alphai), dim3(numunk / numproc / 256), dim3(256));
            double omegar = omega.real();
            double omegai = omega.imag();
            //saxp << <numunk / numproc / 256, 256 >> > (&x_d[addres], &s[addres], &x_d[addres], make_cuDoubleComplex(omegar, omegai));
            saxp(&x_d[addres], &s[addres], &x_d[addres], make_cuDoubleComplex(omegar, omegai), dim3(numunk / numproc / 256), dim3(256));
            //saxp << <numunk / numproc / 256, 256 >> > (&s[addres], &t[addres], &r_d[addres], make_cuDoubleComplex(-1 * omegar, -1 * omegai));
            saxp(&s[addres], &t[addres], &r_d[addres], make_cuDoubleComplex(-1 * omegar, -1 * omegai), dim3(numunk / numproc / 256), dim3(256));
            rnrm2 = sqrt(norm2(r_d));
            error = rnrm2 / bnrm2;
            if (error < res_tol)
                break;
            if (abs(omega) < 1e-200) {
                if (verbos) printf("\t\tBICGSTAB: OMEGA BREAKDOWN\n");
                break;
            }
            rho_1 = rho;
        }
    }
    if (error < res_tol)
    {
        if (verbos) printf("\t\tCONVERGED!\n");
        //else printf("BicGS CONVERGED!...");
    }
    else
    {
        if (verbos) printf("\t\tNOT CONVERGED!****************************iter: % d proc : % d\n", iter, myid);
        //else printf("BicGS NOT CONVERGED...");
    }
    if (verbos) printf("\t\tNUMBER OF %d ITERATIONS: %d (%d)\n", ishermitian, iter, nummatvec);
    if (verbos) printf("\t\tRESIDUAL ERROR NORM: %e\n",error);
    //printf("bicg{iters%d,err%f}...", iter, error);
    printf("\t\tBICGS: NUMBER OF %d ITERATIONS: %d (%d)\n", ishermitian, iter, nummatvec);

            cudaMemcpy(&x[addres], &x_d[addres], numunk / numproc * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}
void FMMFourierTreeGPU::setup_born() {
    int numproc = 1;
    int myid = 0;

    for (int i = 0; i < txproc; i++)
#pragma omp parallel for
        for (int n = 0; n < numunk; n++)
            inc[i * numunk + n] = exp(complex<double>(0, k0 * (tx[mytx + i].real() * pos[n].real() + tx[mytx + i].imag() * pos[n].imag())));

    //generating random input instead of loading phantom.txt
    //generateRandomMeasuredField(); // moved to the beginning of MoM function

    for (int i = 0; i < txproc; i++) {
        fill_n(x, numunk, complex<double>(0, 0));
#pragma omp parallel for
        for (int n = 0; n < numunk; n++)
            g[n] = o[n] * k0 * k0;
        bicgs(x, g, &inc[i * numunk], false, verbosity >= 1);
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

#pragma region redundancy
void FMMFourierTreeGPU::mlfma_redundant(complex<double>* x, complex<double>* r, int redundancyFactor) {
    int numproc = 1;
    int myid = 0;
    int addres = 0;

#if defined(_DEBUG)
#else
    float copyTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    int memory = numunk * sizeof(cuDoubleComplex);
    cudaMemcpy(x_d, x, memory, cudaMemcpyHostToDevice);
    cudaMemcpy(r_d, r, memory, cudaMemcpyHostToDevice);
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&copyTime, start, stop);
    totalCopyTime += (copyTime / 1000);
#endif


    //TRANSFER DATA IN
    //memory = numunk / numproc * sizeof(cuDoubleComplex);
    //addres = myid * numunk / numproc;
    //cudaMemcpy(&x_d[addres], &x[addres], memory, cudaMemcpyHostToDevice);

#if defined(_DEBUG)
#else
    cudaEventRecord(start);
#endif
    mlfma_gpu_redundant(x_d, r_d, redundancyFactor);
    cudaDeviceSynchronize();
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&copyTime, start, stop);
    totalMlfmaTime += (copyTime / 1000);
#endif


#if defined(_DEBUG)
#else
    cudaEventRecord(start);
#endif
    cudaMemcpy(&r[addres], &r_d[addres], memory, cudaMemcpyDeviceToHost);
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&copyTime, start, stop);
    totalCopyTime += (copyTime / 1000);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}
float FMMFourierTreeGPU::nearfield_old(complex<double>* x, complex<double>* r)
{
    float miliseconds;

    int memory = numunk * sizeof(cuDoubleComplex);
#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    cudaMemcpy(x_d, x, memory, cudaMemcpyHostToDevice);
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    totalCopyTime += (miliseconds / 1000);
#endif


#if defined(_DEBUG)
#else
    cudaEventRecord(start);
#endif
    int chunkSize = (int)max(1.0, 8.0 / (double)box * 2 * 4);
    //printf("chunkSize  %d ...", chunkSize);
    //NEARFIELD MULTIPLICATION
    memory = chunkSize * 9 * sizeof(int) + 2 * chunkSize * box * box * sizeof(cuDoubleComplex);
    dim3 P2Pgrid(numclus[level - 1] / chunkSize, 1, 1);
    dim3 P2Pblock(chunkSize * box * box, 1, 1);
    P2Pkernel(r_d, x_d, clusnear_d, near_d, chunkSize, P2Pgrid, P2Pblock, memory, kerrstr);
    cudaDeviceSynchronize();
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    totalComputationTime += (miliseconds / 1000);
#endif


#if defined(_DEBUG)
#else
    cudaEventRecord(start);
#endif
    cudaMemcpy(r, r_d, memory, cudaMemcpyDeviceToHost);
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //totalCopyTime += (miliseconds / 1000);
#endif

    return miliseconds;
}
float FMMFourierTreeGPU::nearfield_modified(complex<double>* x, complex<double>* r)
{
    //in my modification
    //  i solve 2 box per SM becuase i have 128 threads per SM
    //  i wrote kernel for better understanding

    int memory = numunk * sizeof(cuDoubleComplex);
    cudaMemcpy(x_d, x, memory, cudaMemcpyHostToDevice);

    float miliseconds;
#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    //NEARFIELD MULTIPLICATION
    memory = 2 * 9 * sizeof(int) + 8 * box * box * sizeof(cuDoubleComplex); 
    dim3 P2Pgrid(numclus[level - 1] / 2, 1, 1);
    dim3 P2Pblock(2 * box * box, 1, 1);
    P2Pkernel_myModification(r_d, x_d, clusnear_d, near_d, P2Pgrid, P2Pblock, memory, kerrstr);
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
    cudaMemcpy(r, r_d, memory, cudaMemcpyDeviceToHost);
    return miliseconds;
}
void FMMFourierTreeGPU::nearfield_redundant(complex<double>* x, complex<double>* r, int redundancyFactor)
{
    float millisecconds;

    int memory = numunk * sizeof(cuDoubleComplex);

#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    cudaMemcpy(x_d, x, memory, cudaMemcpyHostToDevice);
    cudaMemcpy(r_d, r, memory, cudaMemcpyHostToDevice);
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    totalCopyTime += (millisecconds / 1000);
#endif

    nearfield_redundant(x_d, r_d, redundancyFactor);
}
void FMMFourierTreeGPU::nearfield_redundant(cuDoubleComplex* x_, cuDoubleComplex* r_, int redundancyFactor)
{
    int memory = numunk * sizeof(cuDoubleComplex);
#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    //NEARFIELD MULTIPLICATION
    int chunkSize = (int)max(1.0, 8.0 / (double)box * 2 * 4);
    memory = chunkSize * 9 * sizeof(int) + chunkSize * 2 * box * box * sizeof(cuDoubleComplex);
    //printf("invoking %d blocks...", numclus[level - 1] / chunkSize);
    dim3 P2Pgrid(numclus[level - 1] / chunkSize, 1, 1);
    dim3 P2Pblock(chunkSize * box * box, 1, 1);
    int numRedundantBlocks = redundancyFactor;// (numclus[level - 1] / chunkSize) / redundancyFactor;
    //printf("numRedundantBlocks = %d...", numRedundantBlocks);
    //P2Pkernel_redundancy(r_d, x_d, clusnear_d, near_d, P2Pgrid, P2Pblock, memory, kerrstr);
    P2Pkernel_redundancy(r_, x_, clusnear_d, near_red, numRedundantBlocks, chunkSize, P2Pgrid, P2Pblock, memory, kerrstr);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        printf("\nError executing redundant P2P kernel : %s", cudaGetErrorString(e));


    //complex<double>* checkr = new complex<double>[numunk];
    //cudaMemcpy(checkr, x_, sizeof(cuDoubleComplex) * numunk, cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    //cudaMemcpy(checkr, r_, sizeof(cuDoubleComplex) * numunk, cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
#if defined(_DEBUG)
#else
    float millisecconds;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    totalComputationTime += (millisecconds / 1000);
#endif
    //cudaFree(near_red);
    //cudaMemcpy(r_d, r_, memory, cudaMemcpyDeviceToHost);
}
void FMMFourierTreeGPU::mlfma_gpu_redundant(cuDoubleComplex* x_, cuDoubleComplex* r_, int redundancyFactor)
{
    int numproc = 1;
    int myid = 0;

    float miliseconds;
    int memory;
    int addres;
    int addrss;
    int tile;
    int tilex;
    int tiley;
#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    matvecin++;
    //far field
    mlfma_gpu_farField(x_, r_);
    cudaDeviceSynchronize();
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    totalFarFieldTime += (miliseconds / 1000);

    cudaEventRecord(start);
#endif


    //duplicateData(redundancyFactor);
    nearfield_redundant(x_, r_, redundancyFactor);
    cudaDeviceSynchronize();
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    totalNearFieldTime += (miliseconds / 1000);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}
void FMMFourierTreeGPU::duplicateData(int redundancyFactor)
{
#if defined(_DEBUG)
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float millisecconds = 0.0;
#endif

    //nearfield data duplication
    int redundantBlockSize = 9 * box * box * box * box;
 
    if(near_red == NULL)
    {
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        double redMem = ((((double)redundantBlockSize * redundancyFactor) / 1024.0 / 1024.0) * sizeof(cuDoubleComplex));
        double freMem = ((double)free_memory / 1024.0 / 1024.0);
        //if (verbosity >= 1) 
            printf("for Box=%d, requesting %d MB mem...", box, (int)redMem);
        if (redMem > freMem)
        {
            printf("requesting %d MB mem...", (int)redMem);
            printf("not enough memory on device. //free mem = %d. ignoring test...", (int)freMem);
            return;
        }
        //printf("reqesting for %d Bytes...", redundantBlockSize * redundancyFactor);
        cudaMalloc((void**)&near_red, redundantBlockSize * redundancyFactor * sizeof(cuDoubleComplex));
    }
    

    for (int i = 0; i < redundancyFactor; i++)
    {
        //printf("i %d ", i);
        //printf("copy to index %d out of %d\n", i * redundantBlockSize, redundantBlockSize * redundancyFactor);
        cudaError_t e =
            cudaMemcpy(&near_red[i * redundantBlockSize], near_d, redundantBlockSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        if (e != cudaSuccess)
        {
            printf("\nError copying in redundancy P2P : %s", cudaGetErrorString(e));
            return;
        }
    }
#if defined(_DEBUG)
#else
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    totalDuplicationTime += millisecconds;
#endif
}
int FMMFourierTreeGPU::MoM_redundancy(int redundancyFactor, complex<double>* measuredField)
{
    nummatvec = 0;

    if (measuredField == NULL)
        generateRandomMeasuredField();
    else
        memcpy(o, measuredField, numunk * sizeof(complex<double>));

    //if(verbosity>=1) 
    printf("SETUP BICGS...");
    setup_bicgs();
    //if (verbosity >= 1) 
    printf("SETUP BORN...");
    setup_born();
    //if (verbosity >= 1) 
    printf("SETUP FINISHED...");

    //for redundancy
    duplicateData(redundancyFactor);  //here, must measure data transfer time overhead
    totalCopyTime = 0.0;
    totalDuplicationTime = 0.0;
    totalComputationTime = 0.0;
    totalNearFieldTime = 0.0;
    totalFarFieldTime = 0.0;
    totalMlfmaTime = 0.0;
    vector<complex<double>>vi;
    ot_i.shrink_to_fit();
    ot_i.clear();

    //
    int numproc_mlfma = 1;
    int myid_mlfma = 0;
    int amount = numunk / numproc_mlfma;
    int addres = myid_mlfma * amount;
    //INITIAL GUESS (NO OBJECT)
    if (verbosity >= 1) printf("INITIAL GUESS(NO OBJECT)\n");
    fill_n(o, numunk, complex<double>(0, 0));
    memcpy(tot, inc, numunk * txproc * sizeof(complex<double>));
#pragma omp parallel for
    for (int m = 0; m < numrx * txproc; m++)
        mesd[m] = -mes[m];

    //GRADIENT
    if (verbosity >= 1) printf("GRADIENT\n");
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
        bicgs_redundancy(x, g, c, true, verbosity >= 2, redundancyFactor);
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
    //memcpy(&ot[addres], &od[addres], amount * sizeof(complex<double>));
    memcpy(&ot[addres], &buff[addres], amount * sizeof(complex<double>));
    memcpy(&od[addres], &buff[addres], amount * sizeof(complex<double>));

    if (verbosity >= 1) printf("\n\n**************************** BORN *******************\n");
    for (int iter = 0; iter < bornIteratoins; iter++) {
        printf("Born: iter[%d]...", iter);
        //DENOMINATOR
        if (verbosity >= 1) printf("DENOMINATOR...");
        for (int i = 0; i < txproc; i++) {
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                y[addres + n] = tot[i * numunk + addres + n] * ot[addres + n] * k0 * k0;
            mlfma(y, c);
            fill_n(&x[addres], amount, complex<double>(0, 0));
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                g[addres + n] = o[addres + n] * k0 * k0;
            bicgs_redundancy(x, g, c, false, verbosity >= 2, redundancyFactor);

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
        if (verbosity >= 1) printf("alpha=%3e...", alpha);
        //TAKE STEP
        if (verbosity >= 1) printf("TAKE STEP...");
#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            o[addres + n] = o[addres + n] - alpha * ot[addres + n];

        //UPDATE GREEN'S FUNCTION
        if (verbosity >= 1) printf("UPDATE GREEN'S FUNCTION...");
        for (int i = 0; i < txproc; i++) {
            fill_n(&tot[i * numunk + addres], amount, complex<double>(0, 0));
#pragma omp parallel for
            for (int n = 0; n < amount; n++)
                g[addres + n] = o[addres + n] * k0 * k0;
            bicgs_redundancy(&tot[i * numunk], g, &inc[i * numunk], false, verbosity >= 2, redundancyFactor);
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
        if (verbosity >= 1) printf("GRADIENT...");
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
            bicgs_redundancy(x, g, c, true, verbosity >= 2, redundancyFactor);
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
        if (verbosity >= 1) 
            printf("PR beta=<%3e,%3e>\n", beta.real(), beta.imag());

#pragma omp parallel for
        for (int n = 0; n < amount; n++)
            ot[addres + n] = od[addres + n] + beta * ot[addres + n];

        //or image reconstrungction visualization
        for (int i = 0; i < numunk; i++) vi.push_back(ot[i]);
        ot_i.push_back(vi);
        vi.shrink_to_fit();
        vi.clear();
    }

    return nummatvec;
}
void FMMFourierTreeGPU::bicgs_redundancy(complex<double>* x, complex<double>* o, complex<double>* b, bool ishermitian, bool verbos, int redundancyFactor) 
{
    if (verbos) printf("\nBICGSTAB:\n");
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int numproc_world;
    int myid_world;

    int addres = myid * numunk / numproc;
    cudaMemcpy(&o_d[addres], &o[addres], numunk / numproc * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(&b_d[addres], &b[addres], numunk / numproc * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(&x_d[addres], &x[addres], numunk / numproc * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    int max_it = iti;
    double res_tol = toli;
    complex<double> alpha;
    complex<double> beta;
    complex<double> omega;
    complex<double> rho;
    complex<double> rho_1;

    int iter = 0;
    double rnrm2;
    double snrm2;
    double bnrm2;
    double error;

    bnrm2 = sqrt(norm2(b_d));
    if (bnrm2 < 1e-20)
        bnrm2 = 1;
    //MATVEC 0 START
    matvec_redundancy(x_d, o_d, r_d, ishermitian, redundancyFactor);
    nummatvec++;
    //MATVEC 0 FINISH
    //saxp << <numunk / numproc / 256, 256 >> > (&b_d[addres], &r_d[addres], &r_d[addres], make_cuDoubleComplex(-1, 0));
    saxp(&b_d[addres], &r_d[addres], &r_d[addres], make_cuDoubleComplex(-1, 0), dim3(numunk / numproc / 256), dim3(256));
    //complex<double>* checkr = new complex<double>[100];
    //cudaMemcpy(checkr, r_d, sizeof(complex<double>) * 100, cudaMemcpyDeviceToHost);
    rnrm2 = sqrt(norm2(r_d)); 
    error = rnrm2 / bnrm2;
    //if(myid_world==0)//printf("RES. ERROR: %e ITER: %d\n",error,iter);
    if (error > res_tol) {
        //saxp << <numunk / numproc / 256, 256 >> > (&r_d[addres], &r_d[addres], &r_tld[addres], make_cuDoubleComplex(0, 0));
        saxp(&r_d[addres], &r_d[addres], &r_tld[addres], make_cuDoubleComplex(0, 0), dim3(numunk / numproc / 256), dim3(256));
        //BEGIN ITERATIONS
        while (iter < max_it) {
            if (verbos) printf("\t\tBICGSTAB: iter[%d], error=%f\n", iter, error);
            iter++;
            rho = inner(r_tld, r_d);
            if (abs(rho) < 1e-200) {
                if (verbos) printf("\t\tBICGSTAB: RHO BREAKDOWN\n");
                break;
            }
            if (iter == 1)
                //saxp << <numunk / numproc / 256, 256 >> > (&r_d[addres], &r_d[addres], &p[addres], make_cuDoubleComplex(0, 0));
                saxp(&r_d[addres], &r_d[addres], &p[addres], make_cuDoubleComplex(0, 0), dim3(numunk / numproc / 256), dim3(256));
            else {
                beta = (rho / rho_1) * (alpha / omega);
                double omegar = omega.real();
                double omegai = omega.imag();
                //saxp << <numunk / numproc / 256, 256 >> > (&p[addres], &v[addres], &cbuff[addres], make_cuDoubleComplex(-1 * omegar, -1 * omegai));
                saxp(&p[addres], &v[addres], &cbuff[addres], make_cuDoubleComplex(-1 * omegar, -1 * omegai), dim3(numunk / numproc / 256), dim3(256));
                double betar = beta.real();
                double betai = beta.imag();
                //saxp << <numunk / numproc / 256, 256 >> > (&r_d[addres], &cbuff[addres], &p[addres], make_cuDoubleComplex(betar, betai));
                saxp(&r_d[addres], &cbuff[addres], &p[addres], make_cuDoubleComplex(betar, betai), dim3(numunk / numproc / 256), dim3(256));
            }
            //PRECONDITIONER
            //MATVEC 1 START
            matvec_redundancy(p, o_d, v, ishermitian, redundancyFactor);
            nummatvec++;
            //MATVEC 1 FINISH
            alpha = rho / inner(r_tld, v);
            double alphar = alpha.real();
            double alphai = alpha.imag();
            //saxp << <numunk / numproc / 256, 256 >> > (&r_d[addres], &v[addres], &s[addres], make_cuDoubleComplex(-1 * alphar, -1 * alphai));
            saxp(&r_d[addres], &v[addres], &s[addres], make_cuDoubleComplex(-1 * alphar, -1 * alphai), dim3(numunk / numproc / 256), dim3(256));
            //MATVEC 2 START
            matvec_redundancy(s, o_d, t, ishermitian, redundancyFactor);
            nummatvec++;
            //MATVEC 2 FINISH
            //STABILIZER
            omega = inner(t, s) / norm2(t);
            complex<double> lan1 = inner(t, s);
            double lan2 = norm2(t);
            if (abs(lan1) < 1e-200 && lan2 < 1e-200)omega = 1.0;//PREVENT NAN
            //UPDATE
            //saxp << <numunk / numproc / 256, 256 >> > (&x_d[addres], &p[addres], &x_d[addres], make_cuDoubleComplex(alphar, alphai));
            saxp(&x_d[addres], &p[addres], &x_d[addres], make_cuDoubleComplex(alphar, alphai), dim3(numunk / numproc / 256), dim3(256));
            double omegar = omega.real();
            double omegai = omega.imag();
            //saxp << <numunk / numproc / 256, 256 >> > (&x_d[addres], &s[addres], &x_d[addres], make_cuDoubleComplex(omegar, omegai));
            saxp(&x_d[addres], &s[addres], &x_d[addres], make_cuDoubleComplex(omegar, omegai), dim3(numunk / numproc / 256), dim3(256));
            //saxp << <numunk / numproc / 256, 256 >> > (&s[addres], &t[addres], &r_d[addres], make_cuDoubleComplex(-1 * omegar, -1 * omegai));
            saxp(&s[addres], &t[addres], &r_d[addres], make_cuDoubleComplex(-1 * omegar, -1 * omegai), dim3(numunk / numproc / 256), dim3(256));
            rnrm2 = sqrt(norm2(r_d));
            error = rnrm2 / bnrm2;
            //printf("rnrm2%d...",rnrm2);
            if (error < res_tol)
                break;
            if (abs(omega) < 1e-200) {
                if (verbos) printf("\t\tBICGSTAB: OMEGA BREAKDOWN\n");
                break;
            }
            rho_1 = rho;
        }
    }
    if (error < res_tol)
    {
        if (verbos) printf("\t\tCONVERGED!\n");
        //else printf("BicGS CONVERGED!...");
    }
    else
    {
        if (verbos) printf("\t\tNOT CONVERGED!****************************iter: % d proc : % d\n", iter, myid);
        //else printf("BicGS NOT CONVERGED...");
    }
    if (verbos)
        printf("\t\tNUMBER OF %d ITERATIONS: %d (%d)\n", ishermitian, iter, nummatvec);
    if (verbos) 
        printf("\t\tRESIDUAL ERROR NORM: %e\n", error);
    //printf("bicg{iters%d,err%f}...", iter, error);
    cudaMemcpy(&x[addres], &x_d[addres], numunk / numproc * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}
void FMMFourierTreeGPU::matvec_redundancy(cuDoubleComplex* x, cuDoubleComplex* o, cuDoubleComplex* r, bool ishermitian, int redundancyFactor) 
{
    int numproc = 1;
    int myid = 0;
    //MPI_Comm_size(MPI_COMM_MLFMA, &numproc);
    //MPI_Comm_rank(MPI_COMM_MLFMA, &myid);
    int addres = myid * numunk / numproc;
    if (ishermitian) {
        //prep << <numunk / numproc / 256, 256 >> > (&cbuff[addres], &x[addres]);
        prep(&cbuff[addres], &x[addres], dim3(numunk / numproc / 256), dim3(256));
        mlfma_gpu_redundant(cbuff, r, redundancyFactor);
        //post << <numunk / numproc / 256, 256 >> > (&r[addres], &x[addres], &o[addres]);
        post(&r[addres], &x[addres], &o[addres], dim3(numunk / numproc / 256), dim3(256));
    }
    else {
        //prep << <numunk / numproc / 256, 256 >> > (&cbuff[addres], &x[addres], &o[addres]);
        prep(&cbuff[addres], &x[addres], &o[addres], dim3(numunk / numproc / 256), dim3(256));
        mlfma_gpu_redundant(cbuff, r, redundancyFactor);
        //post << <numunk / numproc / 256, 256 >> > (&r[addres], &x[addres]);
        post(&r[addres], &x[addres], dim3(numunk / numproc / 256), dim3(256));
    }

}
void FMMFourierTreeGPU::generateRandomMeasuredField()
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
#pragma endregion