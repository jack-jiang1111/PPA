// Use "gcc -O3 -fopenmp -o test test.c"
#include <omp.h>
#include <assert.h>
#include <stdio.h>

#define numElems (128*1024*1024)
float src[numElems];
float dest[numElems], destS[numElems];

int main(int argc, char *argv[]){
  int i,j,numThreads;
  double start_time, run_time;

  for (i = 0; i < numElems; i++)
    {
      src[i] = i;
      destS[i] = 0;
      dest[i] = 0;
    }

  //serial version reference
   for (i = 0; i < numElems; ++i)
    destS[i] = src[i]+1;

  numThreads = 8;
  printf("Setting number of threads in parallel region to: %d\n",numThreads);
  omp_set_num_threads(numThreads);

//parallel version 1
  start_time = omp_get_wtime();
#pragma omp parallel for private(i)
  for (i = 0; i < numElems; ++i){
    
    dest[i] = src[i]+1;
  }

  run_time = omp_get_wtime() - start_time;
  printf("Parallel Rate for version 1 is  %.1f GFLOPs \n",numElems*1e-9/run_time);

  //correctness check
  for (i = 0; i < numElems; i++)
    {
      assert(destS[i] == dest[i] );
    }
  printf("Correctness check passed for version 1\n");
//parallel version 2
  start_time = omp_get_wtime();
#pragma omp parallel private(i)
  for (i = 0; i < numElems; ++i){
    
    dest[i] = src[i]+1;
  }

  run_time = omp_get_wtime() - start_time;
  printf("Parallel Rate for version 2 is  %.1f GFLOPs \n",numElems*1e-9/run_time);

  //correctness check
  for (i = 0; i < numElems; i++)
    {
      assert(destS[i] == dest[i] );
    }
  printf("Correctness check passed for version 2\n");
}

