#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#define N 8192
#define Niter 10
#define threshold 0.0000001

void mvseq(int n, double m[][n], double x[n], double y[n]);
void mvpar(int n, double m[][n], double x[n], double y[n]);
void compare(int n, double wref[n], double w[n]);

double A[N][N], x[N],temp[N],xx[N],temp1[N];
int myid, nprocs;
int main(int argc, char *argv[]) {

double clkbegin, clkend;
double t, tmax, *tarr;
int i,j,it;

  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &myid );
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
  tarr = malloc(sizeof(double)*nprocs);

  for(i=0;i<N;i++)
   { 
     x[i] = xx[i] = sqrt(1.0*i);
     for(j=0;j<N;j++) A[i][j] = 2.0*((i+j) % N)/(1.0*N*(N-1));
   }

  if (myid == 0) 
  {
   clkbegin = MPI_Wtime();
   mvseq(N,A,x,temp);
   clkend = MPI_Wtime();
   t = clkend-clkbegin;
   printf("Repeated MV: Sequential Version: Matrix Size = %d; %.2f GFLOPS; Time = %.3f sec; \n",
           N,2.0*1e-9*N*N*Niter/t,t);

  }

  MPI_Barrier(MPI_COMM_WORLD);

  clkbegin = MPI_Wtime();
  mvpar(N,A,xx,temp1);
  clkend = MPI_Wtime();
  t = clkend-clkbegin;
  MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
  
  if (myid == 0)
  {
   printf("Repeated MV: Parallel Version: Matrix Size = %d; %.2f GFLOPS; Time = %.3f sec; \n",
          N,2.0*1e-9*N*N*Niter/tmax,tmax);
   compare(N,x,xx);
  }
  MPI_Finalize();
}


void mvseq(int n, double m[][n], double x[n], double y[n])
{ int i,j,iter;
  
  for(i=0;i<n;i++) { y[i]=0.0; }
  for(iter=0;iter<Niter;iter++)
    {
      for(i=0;i<n;i++){
        for(j=0;j<n;j++){
          y[i] = y[i] + m[i][j]*x[j];
        }
      }
	
      for (i=0; i<N; i++) {
        x[i] = sqrt(y[i]);
        //printf("seq x %d %lf\n",i,x[i]);
      }
    }
}

void mvpar(int n, double m[][n], double x[n], double y[n])
// FIXME: Initially identical to reference; make your changes to parallelize this code
{ 
  int i,j,iter;
  
  //printf("%d\n",sizeof temp_x / sizeof *temp_x);
  for(i=0;i<n;i++) { 
    y[i]=0.0; 
  }
  int ibegin = myid*ceil((float)n/nprocs);
  int iend = (myid+1)*ceil((float)n/nprocs);
  for(iter=0;iter<Niter;iter++){
    double temp_x[n/nprocs];
    MPI_Scatter( x, n/nprocs, MPI_DOUBLE, temp_x, n/nprocs, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);// Broadcast the Vector
    for(i=ibegin;i<iend;i++){
      for(j=0;j<n;j++){
        y[i]+=m[i][j]*x[j];
        //printf("x in the middle %d %lf\n",j,x[j]);
      }
    }
    for(i=ibegin;i<iend;i++){
      temp_x[i-ibegin] = sqrt(y[i]);
      //printf("tempx %d x_copy %d \n",i-ibegin,i);//temp_x[i-ibegin],i,x_copy[i]);
    }
    // Do a sync each iteration
    MPI_Gather( temp_x, n/nprocs, MPI_DOUBLE, x, n/nprocs, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    if(myid==0){
      for(i=0;i<n;i++){
        //printf("x %d %lf\n",i,x[i]);
      }
    }
  }
}

void compare(int n, double wref[n], double w[n])
{
double maxdiff,this_diff;
double minw,maxw,minref,maxref;
int numdiffs;
int i;
  numdiffs = 0;
  maxdiff = 0;
  minw = minref = 1.0e9;
  maxw = maxref = -1.0;
  for (i=0;i<n;i++)
    {
     this_diff = wref[i]-w[i];
     if (w[i] < minw) minw = w[i];
     if (w[i] > maxw) maxw = w[i];
     if (wref[i] < minref) minref = wref[i];
     if (wref[i] > maxref) maxref = wref[i];
     if (this_diff < 0) this_diff = -1.0*this_diff;
     if (this_diff>threshold)
      { numdiffs++;
        printf("%d orginal %lf current %lf\n",i,wref[i],w[i]);
        if (this_diff > maxdiff) maxdiff=this_diff;
      }
    }
   if (numdiffs > 0)
      printf("%d Diffs found over threshold %f; Max Diff = %f\n",
               numdiffs,threshold,maxdiff);
   else
      printf("No differences found between base and test versions\n");
   printf("MinRef = %f; MinPar = %f; MaxRef = %f; MaxPar = %f\n",
          minref,minw,maxref,maxw);
}



/*
module load gcc openmpi
mpicc -O3 -lm -o mat-vec-mul mat-vec-mul.c
mpirun -np 16 ./mat-vec-mul
*/