#include "Test.h"
using namespace std;
#include <fstream>
#include <iomanip>
#include <iostream>
extern int verbosity;

void compareSpeed_CPUDirect(int Level, int BoxPerCluster, int numtests)
{
    //int numtests = 5;

   ////generate tree
   //int BoxPerCluster = 64; //a square number
   //int Level = 10;
    int N = pow(4, Level - 1) * BoxPerCluster;//will be computed based on Level,PoinsPerBox
    FMMFourierTree ftc(N, BoxPerCluster);


    //generate data
    complex<double>* ranIn = new complex<double>[ftc.numunk];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < ftc.numunk; i++)
        ranIn[ftc.unkmap[i]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
    complex<double>* out1 = new complex<double>[ftc.numunk];
    memset(out1, 0, sizeof(complex<double>) * ftc.numunk);
    complex<double>* out2 = new complex<double>[ftc.numunk];
    memset(out2, 0, sizeof(complex<double>) * ftc.numunk);

    //----------------compare speed
    chrono::steady_clock clock;
    chrono::time_point<chrono::steady_clock>a, b;
    double cpuTime, dirTime;
    printf("\ncpu mlfma...");
    a = clock.now();
    for (int i = 0; i < numtests; i++)
        ftc.mlfma(ranIn, out1);
    b = clock.now();
    cpuTime = (chrono::duration_cast<chrono::milliseconds>(b - a).count()) / numtests / 1000.0;
    printf("time = %f milli-secconds\n", cpuTime);
    printf("direct mlfma...");
    a = clock.now();
    for (int i = 0; i < numtests; i++)
        ftc.direct(ranIn, out2);
    b = clock.now();
    dirTime = (chrono::duration_cast<chrono::milliseconds>(b - a).count()) / numtests / 1000.0;
    printf("time = %f milli-secconds\n", dirTime);
    printf("direct mlfma was %s than cpu mlfma, speedup=%f\n", dirTime > cpuTime ? "slower" : "faster", cpuTime / dirTime);

    //--------------compare error
    double tot1 = 0, tot2 = 0, tot3 = 0;
    double err = 0, err2 = 0;
    double maxerr = 0, r1, r2, r3;
    int maxerrind = -1;
    complex<double>tmp;
    for (int i = 0; i < ftc.numunk; i++)
    {
        tmp = out1[i] - out2[i];
        r1 = (pow(tmp.real(), 2) + pow(tmp.imag(), 2));
        tot1 += r1;
        r2 = (pow(out2[i].real(), 2) + pow(out2[i].imag(), 2));
        tot2 += r2;
        tot3 += sqrt((pow(tmp.real(), 2) + pow(tmp.imag(), 2)));

        r3 = (abs(r1 - r2) / r1) * 100;
        if (r3 > maxerr)
        {
            maxerr = r3;
            maxerrind = i;
        }
    }
    err = sqrt(tot1 / tot2);
    err2 = tot3 / ftc.numunk;
    maxerr = sqrt(maxerr);
    printf("\nerr direct mlfma compared with cpu = %f, %f", err, err2);
    printf("\nMax.rel.err = %f%%, were cpu=[%f,%f] and direct=[%f,%f]", maxerr, out1[maxerrind].real(), out1[maxerrind].imag(), out2[maxerrind].real(), out2[maxerrind].imag());
}
void compareSpeed_CPUGPU(int Level, int BoxPerCluster, int numtests)
{
    //int numtests = 5;

   ////generate tree
   //int BoxPerCluster = 64; //a square number
   //int Level = 10;
    int N = pow(4, Level - 1) * BoxPerCluster;//will be computed based on Level,PoinsPerBox
    FMMFourierTreeGPU ftg(N, BoxPerCluster);
    FMMFourierTree ftc(N, BoxPerCluster);


    //generate data
    complex<double>* ranIn = new complex<double>[ftg.numunk];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < ftg.numunk; i++)
        ranIn[ftg.unkmap[i]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
    complex<double>* out1 = new complex<double>[ftg.numunk];
    memset(out1, 0, sizeof(complex<double>) * ftg.numunk);
    complex<double>* out2 = new complex<double>[ftg.numunk];
    memset(out2, 0, sizeof(complex<double>) * ftg.numunk);

    //-----------------------warmup
    printf("\nwarmup...\n");
    for (int i = 0; i < 5; i++)
        ftg.nearfield_old(ranIn, out1);

    //----------------compare speed
    chrono::steady_clock clock;
    chrono::time_point<chrono::steady_clock>a, b;
    double cpuTime, gpuTime;
    printf("\ncpu mlfma...");
    a = clock.now();
    for (int i = 0; i < numtests; i++)
        ftc.mlfma(ranIn, out2);
    b = clock.now();
    cpuTime = (chrono::duration_cast<chrono::milliseconds>(b - a).count()) / numtests / 1000.0;
    printf("time = %f milli-secconds\n", cpuTime);
    printf("gpu mlfma...");
    a = clock.now();
    for (int i = 0; i < numtests; i++)
        ftg.mlfma(ranIn, out1);
    b = clock.now();
    gpuTime = (chrono::duration_cast<chrono::milliseconds>(b - a).count()) / numtests / 1000.0;
    printf("time = %f milli-secconds\n", gpuTime);
    printf("gpu mlfma was %s than cpu mlfma, speedup=%f\n", gpuTime > cpuTime ? "slower" : "faster", cpuTime / gpuTime);

    //--------------compare error
    double tot1 = 0, tot2 = 0, tot3 = 0;
    double err = 0, err2 = 0;
    double maxerr = 0, r1, r2, r3;
    int maxerrind = -1;
    complex<double>tmp;
    memset(out1, 0, ftc.numunk * sizeof(complex<double>));
    memset(out2, 0, ftc.numunk * sizeof(complex<double>));
    ftc.mlfma(ranIn, out1);
    ftg.mlfma(ranIn, out2);
    for (int i = 0; i < ftg.numunk; i++)
    {
        tmp = out1[i] - out2[i];
        r1 = (pow(tmp.real(), 2) + pow(tmp.imag(), 2));
        /*if (isnan(r1))
        {
            r1 = 0;
        }*/
        tot1 += r1;
        /*if (isnan(tot1))
        {
            tot1 = 0;
        }*/
        r2 = (pow(out2[i].real(), 2) + pow(out2[i].imag(), 2));
        tot2 += r2;
        tot3 += sqrt((pow(tmp.real(), 2) + pow(tmp.imag(), 2)));

        r3 = (abs(r1 - r2) / r1) * 100;
        if (r3 > maxerr)
        {
            maxerr = r3;
            maxerrind = i;
        }
    }
    err = sqrt(tot1 / tot2);
    err2 = tot3 / ftg.numunk;
    maxerr = sqrt(maxerr);
    printf("\nerr gpu mlfma compared with cpu = %f, %f", err, err2);
    printf("\nMax.rel.err = %f%%, were cpu=[%f,%f] and gpu=[%f,%f]", maxerr, out1[maxerrind].real(), out1[maxerrind].imag(), out2[maxerrind].real(), out2[maxerrind].imag());
}
void compareErr_GPU_direct(int Level, int BoxPerCluster)
{
    //int BoxPerCluster = 64; //a square number
    //int Level = 5;
    int N = pow(4, Level - 1) * BoxPerCluster;//will be computed based on Level,PoinsPerBox

    FMMFourierTreeGPU ftg(N, BoxPerCluster);

    printf("generating random data...\n");
    complex<double>* ranIn = new complex<double>[ftg.numunk];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < ftg.numunk; i++)
    {
        ranIn[ftg.unkmap[i]] = complex<double>(dis(gen), 0) * complex<double>(0.1, 0);
    }
    complex<double>* out1 = new complex<double>[ftg.numunk];
    memset(out1, 0, sizeof(complex<double>) * ftg.numunk);
    complex<double>* out2 = new complex<double>[ftg.numunk];
    memset(out2, 0, sizeof(complex<double>) * ftg.numunk);

    printf("solution of mlfma gpu...\n");
    ftg.mlfma(ranIn, out1);
    printf("solution of mlfma direct...\n");
    ftg.direct(ranIn, out2);

    //compare error
    double err = 0, err2 = 0;
    double tot1 = 0, tot2 = 0, tot3 = 0;
    double maxerr = 0;
    double prevtot1;
    complex<double>tmp;
    for (int i = 0; i < ftg.numunk; i++)
    {
        tmp = out1[i] - out2[i];
        double r1 = sqrt(pow(tmp.real(), 2) + pow(tmp.imag(), 2));
        prevtot1 = r1;
        tot1 += r1;
        tot2 += sqrt(pow(out2[i].real(), 2) + pow(out2[i].imag(), 2));
        tot3 += sqrt((pow(tmp.real(), 2) + pow(tmp.imag(), 2)));
        if (r1 > maxerr)
            maxerr = r1;
        if (isnan(tot1)) {
            printf("\n\ngoorch!!!  out1=[%f,%f], out2=[%f,%f], tot1=%f, r1=%f, prevtot1=%f", out1[i].real(), out1[i].imag(), out2[i].real(), out2[i].imag(), tot1, r1, prevtot1);
            tot1 = prevtot1;
        }
    }
    err = tot1 / tot2;
    err2 = tot3 / ftg.numunk;
    printf("\nerr redundant kernel compared with non-redundant one = %f, %f", err, err2);
    printf("\nMax err redundant kernel compared with non-redundant one = %f", maxerr);
}

void test_speedMLFMA_redundant(int Level, int BoxPerCluster, int redundancyFactor)
{
    ////generate tree
    //int BoxPerCluster = 64; //a square number
    //int Level = 5;
    int N = pow(4, Level - 1) * BoxPerCluster;//will be computed based on Level,PoinsPerBox
    FMMFourierTreeGPU ftg(N, BoxPerCluster);

    //generate data
    complex<double>* ranIn = new complex<double>[ftg.numunk];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < ftg.numunk; i++)
        ranIn[ftg.unkmap[i]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
    complex<double>* out1 = new complex<double>[ftg.numunk];
    memset(out1, 0, sizeof(complex<double>) * ftg.numunk);
    complex<double>* out2 = new complex<double>[ftg.numunk];
    memset(out2, 0, sizeof(complex<double>) * ftg.numunk);


    //compare speed
    int numtests = 30;
    chrono::steady_clock clock;
    chrono::time_point<chrono::steady_clock> a, b;
    printf("\nnon-redundant kernel...");
    a = clock.now();
    for (int i = 0; i < numtests; i++)
        ftg.mlfma(ranIn, out1);
    b = clock.now();
    double cpuTime = chrono::duration_cast<chrono::milliseconds>(b - a).count() / (double)numtests;
    printf("time = %f milli-secconds\n", cpuTime);
    printf("redundant kernel...");
    a = clock.now();
    for (int i = 0; i < numtests; i++)
        ftg.mlfma_redundant(ranIn, out2, redundancyFactor);
    b = clock.now();
    double gpuTime = chrono::duration_cast<chrono::milliseconds>(b - a).count() / (double)numtests;
    printf("time = %f milli-secconds\n", gpuTime);
    printf("reduntant kernel was %s than non-redundant one, speedup=%f\n", gpuTime > cpuTime ? "slower" : "faster", cpuTime / gpuTime);

    //compare error
    double err = 0, err2 = 0;
    double tot1 = 0, tot2 = 0, tot3 = 0;
    complex<double>tmp;
    for (int i = 0; i < ftg.numunk; i++)
    {
        tmp = out1[i] - out2[i];
        tot1 += (pow(tmp.real(), 2) + pow(tmp.imag(), 2));
        tot2 += (pow(out2[i].real(), 2) + pow(out2[i].imag(), 2));
        tot3 += sqrt((pow(tmp.real(), 2) + pow(tmp.imag(), 2)));
    }
    err = sqrt(tot1 / tot2);
    err2 /= ftg.numunk;
    printf("\nerr redundant kernel compared with non-redundant one = %f, %f", err, err2);
}
void testSpeedP2P_modified(int Level, int BoxPerCluster, int numtests)
{
    //int numtests = 20;

    ////generate tree
    //int BoxPerCluster = 64; //a square number
    //int Level = 8;
    int N = pow(4, Level - 1) * BoxPerCluster;//will be computed based on Level,PoinsPerBox
    //for L=10, PPB=64 -> 16M points
    FMMFourierTreeGPU ftg(N, BoxPerCluster);

    //generate data
    complex<double>* ranIn = new complex<double>[ftg.numunk];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < ftg.numunk; i++)
        ranIn[ftg.unkmap[i]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
    complex<double>* out1 = new complex<double>[ftg.numunk];
    memset(out1, 0, sizeof(complex<double>) * ftg.numunk);
    complex<double>* out2 = new complex<double>[ftg.numunk];
    memset(out2, 0, sizeof(complex<double>) * ftg.numunk);

    //-----------------------warmup
    printf("\nwarmup...\n");
    for (int i = 0; i < 5; i++)
        ftg.nearfield_old(ranIn, out1);

    //----------------compare speed
    float oldTime, modTime;
    printf("modified kernel...");
    for (int i = 0; i < numtests; i++)
        modTime = ftg.nearfield_modified(ranIn, out2);
    printf("time = %f milli-secconds\n", modTime);
    printf("old kernel...");
    for (int i = 0; i < numtests; i++)
        oldTime = ftg.nearfield_old(ranIn, out1);
    printf("time = %f milli-secconds\n", oldTime);
    printf("reduntant kernel was %s than old one, speedup=%f\n", modTime > oldTime ? "slower" : "faster", oldTime / modTime);

    //--------------compare error
    double err = 0, err2 = 0;
    double tot1 = 0, tot2 = 0, tot3 = 0;
    complex<double>tmp;
    for (int i = 0; i < ftg.numunk; i++)
    {
        tmp = out1[i] - out2[i];
        tot1 += (pow(tmp.real(), 2) + pow(tmp.imag(), 2));
        tot2 += (pow(out2[i].real(), 2) + pow(out2[i].imag(), 2));
        tot3 += sqrt((pow(tmp.real(), 2) + pow(tmp.imag(), 2)));
    }
    err = sqrt(tot1 / tot2);
    err2 = tot3 / ftg.numunk;
    printf("\nerr modified kernel compared with old one = %f, %f", err, err2);
}
void testSpeedP2P_redundant(int Level, int BoxPerCluster, int numtests, int redundancyFactor)
{
    //int numtests = 5;

    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> a, b;
    int time;

    ////generate tree
    //int BoxPerCluster = 64; //a square number
    //int Level = 10;
    int N = pow(4, Level - 1) * BoxPerCluster;//will be computed based on Level,PoinsPerBox
    FMMFourierTreeGPU ftg(N, BoxPerCluster);
    ftg.duplicateData(redundancyFactor);

    //generate data
    complex<double>* ranIn = new complex<double>[ftg.numunk];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < ftg.numunk; i++)
        ranIn[ftg.unkmap[i]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
    complex<double>* out1 = new complex<double>[ftg.numunk];
    memset(out1, 0, sizeof(complex<double>) * ftg.numunk);
    complex<double>* out2 = new complex<double>[ftg.numunk];
    memset(out2, 0, sizeof(complex<double>) * ftg.numunk);

    //-----------------------warmup
    printf("\nwarmup...\n");
    for (int i = 0; i < 5; i++)
        ftg.nearfield_old(ranIn, out1);

    //----------------compare speed
    double newTime, oldTime;
    printf("\nredundant kernel...");
    a = clock.now();
    for (int i = 0; i < numtests; i++)
        ftg.nearfield_redundant(ranIn, out2, redundancyFactor);
    b = clock.now();
    newTime = ((double)(chrono::duration_cast<chrono::microseconds>(b - a).count()) / 1000) / numtests;
    printf("time = %f milli-secconds\n", newTime);
    printf("non-redundant kernel...");
    for (int i = 0; i < numtests; i++)
        oldTime = ftg.nearfield_old(ranIn, out1);
    printf("time = %f milli-secconds\n", oldTime);
    printf("reduntant kernel was %s than non-redundant one, speedup=%f\n", newTime > oldTime ? "slower" : "faster", oldTime / newTime);

    //--------------compare error
    double tot1 = 0, tot2 = 0, tot3 = 0;
    double err = 0, err2 = 0;
    double maxerr = 0, r1, r2, r3;
    int maxerrind = -1;
    complex<double>tmp;
    for (int i = 0; i < ftg.numunk; i++)
    {
        tmp = out1[i] - out2[i];
        r1 = (pow(tmp.real(), 2) + pow(tmp.imag(), 2));
        tot1 += r1;
        r2 = (pow(out1[i].real(), 2) + pow(out1[i].imag(), 2));
        tot2 += r2;
        tot3 += sqrt((pow(tmp.real(), 2) + pow(tmp.imag(), 2)));

        r3 = (abs(r1 - r2) / r1) * 100;
        if (r3 > maxerr)
        {
            maxerr = r3;
            maxerrind = i;
        }
    }
    err = sqrt(tot1 / tot2);
    err2 = tot3 / ftg.numunk;
    maxerr = sqrt(maxerr);
    printf("\nerr redundant kernel compared with non-redundant one = %f, %f", err, err2);
    printf("\nMax.rel.err = %f%%, were redundant=[%f,%f] and non-redundant=[%f,%f]", maxerr, out1[maxerrind].real(), out1[maxerrind].imag(), out2[maxerrind].real(), out2[maxerrind].imag());
}

void testSpeedP2P_byLevel(int N_, int numtests)
{
    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> a, b;
    int time;

    int Box = 8;
    int Level = log(N_ / Box * Box) / log(4);
    
    for (int levelinc = 1; levelinc <= 2; levelinc++)
    {
        int BoxPerCluster = pow(Box * pow(2, -levelinc), 2);
        int level = max(4, Level + levelinc);
        int N = pow(4, level) * BoxPerCluster;//will be computed based on Level,PoinsPerBox
        
        printf("N=%d, L=%d[inc%d], t=%d, P2P kernel...", N, Level, levelinc, BoxPerCluster);
        FMMFourierTreeGPU* ftg = new FMMFourierTreeGPU(N, BoxPerCluster);

        //generate data
        complex<double>* ranIn = new complex<double>[ftg->numunk];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < ftg->numunk; i++)
            ranIn[ftg->unkmap[i]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
        complex<double>* out = new complex<double>[ftg->numunk];

        //warm up
        ftg->nearfield_old(ranIn, out);

        //----------------invoke
        a = clock.now();
        for (int i = 0; i < numtests; i++)
            ftg->nearfield_old(ranIn, out);
        b = clock.now();
        double rep = (double)(chrono::duration_cast<chrono::microseconds>(b - a).count()) / numtests / 1000;
        printf("time = %f milli-secconds\n", rep);

        ftg->~FMMFourierTreeGPU();//deallocate memory
        //delete ftg;
    }
}
void testSpeedP2P_byLevel_redundant(int N_, int numtests, int redundancyFactor)
{
    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> a, b;
    int time;

    int Box = 8;
    int Level = log(N_ / Box * Box) / log(4);

    for (int levelinc = -1; levelinc <= 1; levelinc++)
    {
        int BoxPerCluster = pow(Box * pow(2, -levelinc), 2);
        int level = max(4, Level + levelinc);
        int N = pow(4, level) * BoxPerCluster;//will be computed based on Level,PoinsPerBox

        printf("N=%d, L=%d[inc%d], t=%d, P2P kernel...", N, Level, levelinc, BoxPerCluster);
        FMMFourierTreeGPU* ftg = new FMMFourierTreeGPU(N, BoxPerCluster);
        ftg->duplicateData(redundancyFactor);

        //generate data
        complex<double>* ranIn = new complex<double>[ftg->numunk];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < ftg->numunk; i++)
            ranIn[ftg->unkmap[i]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
        complex<double>* out = new complex<double>[ftg->numunk];

        //warm up
        ftg->nearfield_old(ranIn, out);

        //----------------invoke
        a = clock.now();
        for (int i = 0; i < numtests; i++)
            ftg->nearfield_redundant(ranIn, out, redundancyFactor);
        b = clock.now();
        double rep = (double)(chrono::duration_cast<chrono::microseconds>(b - a).count()) / numtests / 1000;
        printf("time = %f milli-secconds\n", rep / numtests);

        ftg->~FMMFourierTreeGPU();//deallocate memory
        //delete ftg;
    }
}
void testSpeedAlgorithm_byLevel(int N_, int numtests)
{
    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> a, b;
    int time;

    int Box = 8;
    int Level = log(N_ / Box * Box) / log(4);

    for (int levelinc = -1; levelinc <= 1; levelinc++)
    {
        int BoxPerCluster = pow(Box * pow(2, -levelinc), 2);
        int level = max(4, Level + levelinc);
        int N = pow(4, level) * BoxPerCluster;//will be computed based on Level,PoinsPerBox

        printf("N=%d, L=%d, inc=%d, t=%d,", N, Level, levelinc, BoxPerCluster);
        FMMFourierTreeGPU* ftg = new FMMFourierTreeGPU(N, BoxPerCluster);

        //generate data
        complex<double>* ranIn = new complex<double>[ftg->numunk];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < ftg->numunk; i++)
            ranIn[ftg->unkmap[i]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
        complex<double>* out = new complex<double>[ftg->numunk];

        ////warm up
        //ftg->nearfield_old(ranIn, out);

        //----------------invoke
        a = clock.now();
        for (int i = 0; i < numtests; i++)
        {
            //printf("test%d...", i);
            ftg->mlfma(ranIn, out);
        }
        b = clock.now();
        double rep = (double)(chrono::duration_cast<chrono::microseconds>(b - a).count()) / numtests / 1000;
        printf("\n\tavg time = %f milli-secconds,", rep);
        printf("\n\tMLFMA_time = %f, nearFieldTime = %f, farFieldTime = %f,",
            ftg->totalMlfmaTime / numtests,
            ftg->totalNearFieldTime / numtests,
            ftg->totalFarFieldTime / numtests);
        printf("\n\ttransferTime = %f, computationTime = %f\n",
            ftg->totalCopyTime / numtests,
            ftg->totalComputationTime / numtests);

        ftg->~FMMFourierTreeGPU();//deallocate memory
        //delete ftg;
    }
}
void testSpeedAlgorithm_byLevel_redundant(int N_, int numtests, int redundancyFactor)
{
    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> a, b;
    int time;

    int Box = 8;
    int Level = log(N_ / Box * Box) / log(4);

    for (int levelinc = -1; levelinc <= 1; levelinc++)
    {
        int BoxPerCluster = pow(Box * pow(2, -levelinc), 2);
        int level = max(4, Level + levelinc);
        int N = pow(4, level) * BoxPerCluster;//will be computed based on Level,PoinsPerBox

        printf("N=%d, L=%d, inc=%d, t=%d, redundancyFactor=%d,", N, Level, levelinc, BoxPerCluster, redundancyFactor);
        FMMFourierTreeGPU* ftg = new FMMFourierTreeGPU(N, BoxPerCluster);
        ftg->duplicateData(redundancyFactor);

        //generate data
        complex<double>* ranIn = new complex<double>[ftg->numunk];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < ftg->numunk; i++)
            ranIn[ftg->unkmap[i]] = complex<double>(dis(gen) * 10, 0) * complex<double>(0.1, 0);
        complex<double>* out = new complex<double>[ftg->numunk];

        ////warm up
        //ftg->nearfield_old(ranIn, out);

        //----------------invoke
        double rep;
        a = clock.now();
        for (int i = 0; i < numtests; i++)
        {
            //printf("test%d...", i);
            ftg->mlfma_redundant(ranIn, out, redundancyFactor);
        }
        b = clock.now();
        rep = ((double)(chrono::duration_cast<chrono::microseconds>(b - a).count()) / 1000) / numtests;
        printf("\n\tavg time = %f milli-secconds,", rep);
        printf("\n\tMLFMA_time = %f, nearFieldTime = %f, farFieldTime = %f,",
            ftg->totalMlfmaTime / numtests,
            ftg->totalNearFieldTime / numtests,
            ftg->totalFarFieldTime / numtests);
        printf("\n\ttransferTime = %f, duplicationTime = %f, computationTime = %f\n",
            ftg->totalCopyTime / numtests,
            ftg->totalDuplicationTime / numtests,
            ftg->totalComputationTime / numtests);

        ftg->~FMMFourierTreeGPU();//deallocate memory
        //delete ftg;
    }
}

void testSpeedDBIM(int N_, int numtests)
{
    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> a, b;
    int time;

    int Box = 8;
    int Level = log(N_ / Box * Box) / log(4);

    for (int levelinc = 1; levelinc <= 1; levelinc++)
    {
        int BoxPerCluster = pow(Box * pow(2, -levelinc), 2);
        int level = max(4, Level + levelinc);
        int N = pow(4, level) * BoxPerCluster;//will be computed based on Level,PoinsPerBox

        printf("N=%d, L=%d[inc%d], t=%d, MLFMA ...", N, Level, levelinc, BoxPerCluster);
        FMMFourierTreeGPU* ftg = new FMMFourierTreeGPU(N, BoxPerCluster);

        //----------------invoke
        double rep;
        int numMLFMAcalls = 0;
        a = clock.now();
        for (int i = 0; i < numtests; i++)
        {
            numMLFMAcalls += ftg->MoM();
        }
        b = clock.now();
        rep = ((double)(chrono::duration_cast<chrono::microseconds>(b - a).count()) / 1000) / numtests;
        printf("time = %f milli-secconds, while invoked MLFMA %d times\n", rep, numMLFMAcalls/numtests);

        ftg->~FMMFourierTreeGPU();//deallocate memory
        //delete ftg;
    }
}
void testSpeedDBIM_redundant(int N_, int numtests, int redundancyFactor)
{
    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> a, b;
    int time;

    int Box = 8;
    int Level = log(N_ / Box * Box) / log(4);

    for (int levelinc = -1; levelinc <= 1; levelinc++)
    {
        int BoxPerCluster = pow(Box * pow(2, -levelinc), 2);
        int level = max(4, Level + levelinc);
        int N = pow(4, level) * BoxPerCluster;

        printf("N=%d, L=%d[inc%d], t=%d, MLFMA ...", N, Level, levelinc, BoxPerCluster);
        FMMFourierTreeGPU* ftg = new FMMFourierTreeGPU(N, BoxPerCluster);

        //----------------invoke
        double rep;
        int numMLFMAcalls = 0;
        a = clock.now();
        for (int i = 0; i < numtests; i++)
        {
            //printf("test%d...", i);
            numMLFMAcalls += ftg->MoM_redundancy(redundancyFactor);
        }
        b = clock.now();
        rep = ((double)(chrono::duration_cast<chrono::microseconds>(b - a).count()) / 1000);
        printf("DBIM avg execution time = %f milli-secconds, average of total times:", 
            rep / numtests);
        printf("\n\ttrnasfer_time = %f, kernel_time = %f,", 
            ftg->totalCopyTime / numtests, ftg->totalComputationTime / numtests);
        printf("\n\tnearFieldTime = %f, farFieldTime = %f, while invoked MLFMA% d times in each step\n",
            ftg->totalFarFieldTime, ftg->totalNearFieldTime, numMLFMAcalls / numtests);

        //writeComplexVectorsToFile(ftg->ot_i, 0);

        ftg->~FMMFourierTreeGPU();//deallocate memory
        //delete ftg;
    }
}

void compareSpeedDBIM_finalObject(int N_, int levelinc, int bornIteratoins, int redundancyFactor)
{
    int Box = 8;
    int N = N_ * Box * Box;
    int Level = log(N / Box * Box) / log(4);

        int BoxPerCluster = pow(Box * pow(2, -levelinc), 2);
        //int level = max(4, Level + levelinc);
        //int N = pow(4, level) * BoxPerCluster;//will be computed based on Level,PoinsPerBox
        //generate random data as measured field
        

        printf("N=%d, L=%d[inc%d], t=%d, MLFMA ...", N, Level, levelinc, BoxPerCluster);
        FMMFourierTreeGPU* ftg = new FMMFourierTreeGPU(N, BoxPerCluster);
        printf("numunk=%d...", ftg->numunk);
        ftg->generateRandomMeasuredField(); //generate random measured field
        vector<complex<double>> randomMeasuredField = ftg->generatedField;
        ftg->bornIteratoins = bornIteratoins;
        ftg->MoM(randomMeasuredField.data());
        writeComplexVectorsToFile(ftg->ot_i, 0);
        ftg->~FMMFourierTreeGPU();//deallocate memory
        //delete ftg;


        printf("N=%d, L=%d[inc%d], t=%d, MLFMA redundant ...", N, Level, levelinc, BoxPerCluster);
        ftg = new FMMFourierTreeGPU(N, BoxPerCluster);
        ftg->bornIteratoins = bornIteratoins;
        ftg->MoM_redundancy(redundancyFactor, randomMeasuredField.data());//give the same initial measured field
        writeComplexVectorsToFile(ftg->ot_i, redundancyFactor);
        ftg->~FMMFourierTreeGPU();//deallocate memory
        //delete ftg;
}
void writeComplexVectorsToFile(const vector<vector<complex<double>>>& data, int redundancyFactor) {
    for (size_t i = 0; i < data.size(); ++i) {
        string filename = "image_" + to_string(i) + 
            (redundancyFactor>0 ? ("_redfac" + to_string(redundancyFactor)) : "") + ".txt"; // Create a filename for each vector
        ofstream outFile(filename);

        if (!outFile) {
            cerr << "Error opening file: " << filename << endl;
            continue;
        }

        for (const auto& c : data[i]) {
            outFile << fixed << setprecision(6) << c.real() << " " << c.imag() << endl; // Write real and imaginary parts
        }

        outFile.close();
    }
}

void test_speedBICGS_MLFAM_GPU(int N_, int numtests)
{
    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> a, b;
    int time;

    int Box = 8;
    int Level = log(N_ / Box * Box) / log(4);

        int BoxPerCluster = pow(Box * pow(2, -0), 2);
        int level = max(4, Level + 0);
        int N = pow(4, level) * BoxPerCluster;//will be computed based on Level,PoinsPerBox

        printf("N=%d, L=%d[inc%d], t=%d, MLFMA ...", N, Level, 0, BoxPerCluster);
        FMMFourierTreeGPU* ftg = new FMMFourierTreeGPU(N, BoxPerCluster);
        ftg->setup_bicgs();
        ftg->generateRandomMeasuredField(); //generate random measured field
        vector<complex<double>> randomMeasuredField = ftg->generatedField;
        memcpy(ftg->o, randomMeasuredField.data(), ftg->numunk * sizeof(complex<double>));
        for (int n = 0; n < ftg->numunk; n++)
            ftg->inc[n] = exp(complex<double>(0, ftg->k0 * (ftg->tx[0].real() * ftg->pos[n].real() + ftg->tx[0].imag() * ftg->pos[n].imag())));
        ftg->iti = 30;//bicgs max iterations

        //----------------invoke
        double rep;
        int numMLFMAcalls = 0;
        for (int i = 0; i < numtests; i++)
        {
            a = clock.now();
            ftg->bicgs(ftg->x, ftg->o, ftg->inc, false, false);
            b = clock.now();
            rep += ((double)(chrono::duration_cast<chrono::microseconds>(b - a).count()) / 1000) / numtests;
            fill_n(ftg->x, ftg->numunk, complex<double>(0, 0));
        }
        printf("time = %f milli-secconds, while invoked MLFMA %d times\n", rep, numMLFMAcalls / numtests);

        ftg->~FMMFourierTreeGPU();//deallocate memory
        //delete ftg;
}
void test_speedBICGS_MLFAM_CPU(int N_, int numtests)
{
    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> a, b;
    int time;

    int Box = 8;
    int Level = log(N_ / Box * Box) / log(4);

        int BoxPerCluster = pow(Box * pow(2, -0), 2);
        int level = max(4, Level + 0);
        int N = pow(4, level) * BoxPerCluster;//will be computed based on Level,PoinsPerBox

        printf("N=%d, L=%d[inc%d], t=%d, MLFMA ...", N, Level, 0, BoxPerCluster);
        FMMFourierTree* ftg = new FMMFourierTree(N, BoxPerCluster);
        ftg->setup_bicgs();
        ftg->generateRandomMeasuredField(); //generate random measured field
        vector<complex<double>> randomMeasuredField = ftg->generatedField;
        memcpy(ftg->o, randomMeasuredField.data(), ftg->numunk * sizeof(complex<double>));
        for (int n = 0; n < ftg->numunk; n++)
            ftg->inc[n] = exp(complex<double>(0, ftg->k0 * (ftg->tx[0].real() * ftg->pos[n].real() + ftg->tx[0].imag() * ftg->pos[n].imag())));
        ftg->iti = 30;//bicgs max iterations

        //----------------invoke
        double rep;
        int numMLFMAcalls = 0;
        for (int i = 0; i < numtests; i++)
        {
            a = clock.now();
            ftg->bicgs(ftg->x, ftg->o, ftg->inc, false);
            b = clock.now();
            rep += ((double)(chrono::duration_cast<chrono::microseconds>(b - a).count()) / 1000) / numtests;
            fill_n(ftg->x, ftg->numunk, complex<double>(0, 0));
        }
        printf("time = %f milli-secconds, while invoked MLFMA %d times\n", rep, numMLFMAcalls / numtests);

        ftg->~FMMFourierTree();//deallocate memory
        //delete ftg;
}

