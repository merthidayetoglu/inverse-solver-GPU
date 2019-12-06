#!/bin/bash

#PBS -l nodes=64:ppn=16:xk
#PBS -l walltime=00:05:00
##PBS -l advres=wenmeixe
#PBS -q debug
cd $PBS_O_WORKDIR

export NUMFREQ=1
export MAXFREQ=1
export MINFREQ=0.25

export NUMRX=1024
export NUMTX=1024
export TXPROC=16

export MLFMA=1
export OMP_NUM_THREADS=8
aprun -n 64 -N 1 -d $OMP_NUM_THREADS -j 1 ./mom
