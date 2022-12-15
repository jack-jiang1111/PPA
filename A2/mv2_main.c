// Use "gcc -O3 -fopenmp" to compile

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#ifndef N
#define N (4096)
#endif
#define NTrials (5)
#define threshold (0.0000001)

double A[N][N], x[N], y[N], z[N], yy[N], zz[N];
void compare(int n, double wref[n], double w[n]);
void mv2_seq(int n, double *__restrict__ m, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z);
void mv2_par(int n, double *__restrict__ m, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z);


int main(int argc, char *argv[]){
  double tstart,telapsed;
  
  int i,j,k,nt,trial,max_threads,num_cases;
  int nthr_32[9] = {1,2,4,8,10,12,14,15,31};
  int nthr_40[9] = {1,2,4,8,10,12,14,19,39};
  int nthr_48[9] = {1,2,4,8,10,15,20,23,47};
  int nthr_56[9] = {1,2,4,8,10,15,20,27,55};
  int nthr_64[9] = {1,2,4,8,10,15,20,31,63};
  int nthreads[9];
  double mint_par[9],maxt_par[9];
  double mint_seq,maxt_seq;
  
  printf("Matrix Size = %d; NTrials=%d\n",N,NTrials);
  
    for (i = 0; i < N; i++) {
    x[i] = i;
    for (j = 0; j < N; j++)
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
  }
  
  printf("Reference sequential code performance in GFLOPS");
  mint_seq = 1e9; maxt_seq = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   for (i = 0; i < N; i++) { y[i] = 0; z[i] = 0.0; }
   tstart = omp_get_wtime();
   mv2_seq(N, &A[0][0], x, y, z);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint_seq) mint_seq=telapsed;
   if (telapsed > maxt_seq) maxt_seq=telapsed;
  }
  printf(" Min: %.2f; Max: %.2f\n",4.0e-9*N*N/maxt_seq,4.0e-9*N*N/mint_seq);
  
  max_threads = omp_get_max_threads();
  printf("Max Threads (from omp_get_max_threads) = %d\n",max_threads);
  switch (max_threads)
  {
	  case 32: for(i=0;i<9;i++) nthreads[i] = nthr_32[i]; num_cases=9; break;
	  case 40: for(i=0;i<9;i++) nthreads[i] = nthr_40[i]; num_cases=9; break;
	  case 48: for(i=0;i<9;i++) nthreads[i] = nthr_48[i]; num_cases=9; break;
	  case 56: for(i=0;i<9;i++) nthreads[i] = nthr_56[i]; num_cases=9; break;
	  case 64: for(i=0;i<9;i++) nthreads[i] = nthr_64[i]; num_cases=9; break;
	  default: {
                    nt = 1;i=0;
                    while (nt <= max_threads) {nthreads[i]=nt; i++; nt *=2;}
                    if (nthreads[i-1] < max_threads) {nthreads[i] = max_threads; i++;}
                    num_cases = i;
                    nthreads[num_cases-1]--;
                    nthreads[num_cases-2]--;
		   }
  }

  for (nt=0;nt<num_cases;nt ++)
  {
   omp_set_num_threads(nthreads[nt]);
   mint_par[nt] = 1e9; maxt_par[nt] = 0;
   for (trial=0;trial<NTrials;trial++)
   {
    for (i = 0; i < N; i++) { yy[i] = 0; zz[i] = 0.0; }
    tstart = omp_get_wtime();
    mv2_par(N, &A[0][0], x, yy, zz);
    telapsed = omp_get_wtime()-tstart;
    if (telapsed < mint_par[nt]) mint_par[nt]=telapsed;
    if (telapsed > maxt_par[nt]) maxt_par[nt]=telapsed;
   }
  compare(N, y, yy);
  compare(N, z, zz); 
  }
  printf("Performance (Best & Worst) of parallelized version: GFLOPS on ");
  for (nt=0;nt<num_cases-1;nt++) printf("%d/",nthreads[nt]);
  printf("%d threads\n",nthreads[num_cases-1]);
  printf("Best Performance (GFLOPS): ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",4.0e-9*N*N/mint_par[nt]);
  printf("\n");
  printf("Worst Performance (GFLOPS): ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",4.0e-9*N*N/maxt_par[nt]);
  printf("\n");
}

void compare(int n, double wref[n], double w[n]) {
  double maxdiff, this_diff;
  int numdiffs;
  int i;
  numdiffs = 0;
  maxdiff = 0;
  for (i = 0; i < n; i++) {
    this_diff = wref[i] - w[i];
    if (this_diff < 0)
      this_diff = -1.0 * this_diff;
    if (this_diff > threshold) {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }
  if (numdiffs > 0)
    printf("Correctness Check Failed: %d Diffs found over threshold %f; Max Diff = %f\n", numdiffs,
           threshold, maxdiff);
}
