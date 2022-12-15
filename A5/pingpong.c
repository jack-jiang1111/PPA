#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include "mpi.h"
#define TAG 0

int main(int argc, char* argv[]){
  double clockBegin, clockEnd, t=0;
  double *sBuff, *rBuff;
  int arrCount[] = {1, 8, 64, 512, 4096, 32768, 262144,1048576};
  int rank, count,msgLen,maxMsgLen;        

  maxMsgLen = arrCount[7];
  sBuff = (double *) malloc(maxMsgLen*sizeof(double));
  for(int i=0;i<maxMsgLen;i++) sBuff[i] = i;
  rBuff = (double *) malloc(maxMsgLen*sizeof(double));

  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for(int i =0;i<8;i++){
   count = arrCount[i];
   int nIter = 100000000 / (100 + count);
   if(rank== 0) clockBegin = MPI_Wtime();
   for(int i=0;i <nIter;i++){
// Use blocking send recv
   if (rank==0){
    //
    MPI_Send(sBuff, count, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD);
    MPI_Recv(rBuff, count, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD, &status);
    MPI_Send(rBuff, count, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD);
    MPI_Recv(sBuff, count, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD, &status);
   }
   else if (rank==1){
    //
    MPI_Recv(sBuff, count, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Send(sBuff, count, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
    MPI_Recv(rBuff, count, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &status);
    MPI_Send(rBuff, count, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
   }
  }
  if(rank == 0){
    clockEnd = MPI_Wtime();
    t = (clockEnd - clockBegin)/(4*nIter);
    printf("Message length: %d , GBytes/sec: %.3f , NIters: %d , Time taken per message (microsecs): %.3f , Total time (secs): %.3f\n",count, count * sizeof(double) * 1e-9 * 4 * nIter  / (clockEnd- clockBegin),nIter, 1e6*t, clockEnd - clockBegin);
   }
  }
  free(sBuff);
  free(rBuff);
  MPI_Finalize();
  return 0;
}

