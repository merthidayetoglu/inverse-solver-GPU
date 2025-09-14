#include "Test.h"
using namespace std;

int verbosity = 0;

int main()
{
    //compareSpeed_CPUDirect(5, 64, 1);
    //compareSpeed_CPUGPU(6, 64, 1);
    //compareErr_GPU_direct(5, 64);

    ////in my modification:
    ////  i solved 2 boxes per block instead of 4, because my GPU has 128 threads (8^2 boxes per cluster -> 2 clusters per SM)
    //testSpeedP2P_modified(7, 64, 10);

    //in redundant kernel:
    //  i duplicate near array for each 16 blocks (each 32 cluster)
    //testSpeedP2P_redundant(6, 64, 30, 1024);

    ////redundant kernl, different densities and problem sizes
    //// this only runs a single MLFMA
    ////some times there is an unkown memory access error which solves by re-running
    //// to solve, you must first execute it for inc=0,1 and once again for inc=-1
    //for(int i = 6; i<= 8; i++) testSpeedAlgorithm_byLevel_redundant(pow(4, i), 20, 2);
    //for(int i = 5; i<= 9; i++) testSpeedAlgorithm_byLevel_redundant(pow(4, i), 50, 2);
    for(int i = 5; i <= 9; i++) testSpeedAlgorithm_byLevel(pow(4, i), 10);
    //for(int i = 4; i <= 6; i++) testSpeedAlgorithm_byLevel(pow(4, i), 50);

    ////test redundancy for
    ////  acceleration of DBIM with random initial data
    ////  and for less dense tree
   //for (int i = 6; i <= 8; i++) testSpeedDBIM(pow(4, i), 3);
   //for (int i = 6; i <= 8; i++) testSpeedDBIM_redundant(pow(4, i), 3, 2);

    //testSpeedDBIM_redundant(pow(4, 7), 3, 2);  //why it never ends? it is too slow for problems larger than 4^5
    //testSpeedDBIM(pow(4, 7), 3);
    ////testSpeedDBIM_redundant(pow(4, 9), 3, 2);
    ////testSpeedDBIM(pow(4, 9), 3);

    ///writing pictures to file
    //compareSpeedDBIM_finalObject(pow(4, 5), 0, 30, 2);


    //for (int i = 6; i <= 10; i++)
    //    test_speedBICGS_MVP_CPU(pow(2, i), 3);
    //for (int i = 8; i <= 20; i++)
    //    test_speedBICGS_MLFAM_CPU(pow(2, i), 3);
    //for(int i = 16; i <=20; i++)
    //    test_speedBICGS_MLFAM_GPU(pow(2, i), 3);

    return 0;
}
