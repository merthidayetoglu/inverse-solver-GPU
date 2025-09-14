#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <stdio.h>

#include "FMMFourierTree.h"
#include "FMMFourierTreeGPU.cuh"


//test code of mert
void compareSpeed_CPUDirect(int Level, int PointsPerBox, int numtests);
void compareSpeed_CPUGPU(int Level, int PointsPerBox, int numtests); //speed of CPU and GPU versions of mert implementation
void compareErr_GPU_direct(int Level, int PointsPerBox); //err of mert GPU implementation w.r.t direct method

//test code of mert : density
void testSpeedP2P_byLevel(int N, int numtests);
void testSpeedP2P_byLevel_redundant(int N, int numtests, int redundancyFactor);
void testSpeedAlgorithm_byLevel(int N_, int numtests);
void testSpeedAlgorithm_byLevel_redundant(int N_, int numtests, int redundancyFactor);


//test my improvements
void test_speedMLFMA_redundant(int Level, int PointsPerBox, int redundancyFactor); //test speed of whole mlfma with redundant kernel
void testSpeedP2P_modified(int Level, int PointsPerBox, int numTests); //mikro kernel test, my modification on mert's kernel
void testSpeedP2P_redundant(int Level, int PointsPerBox, int numTests, int redundancyFactor); //mikro kernel test, redundancy vs mert

//test speed of DBIM
void testSpeedDBIM(int N_, int numtests);
void testSpeedDBIM_redundant(int N_, int numtests, int redundancyFactor);

void writeComplexVectorsToFile(const vector<vector<complex<double>>>& data, int redundancyFactor);
void compareSpeedDBIM_finalObject(int N_, int levelinc, int bornIteratoins, int redundancyFactor);

//test speed of BiCGS
void test_speedBICGS_MLFAM_GPU(int N_, int numtests);
void test_speedBICGS_MLFAM_CPU(int N_, int numtests);
void test_speedBICGS_MVP_CPU(int N_, int numtests);
