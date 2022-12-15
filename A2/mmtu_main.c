// Use "gcc -O3 -fopenmp" to compile

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#ifndef N
#define N (801)
#endif
#define NTrials (5)
#define threshold (0.0000001)

void compare(int n, double wref[][n], double w[][n], int numt);
void mmtu_seq(int n, double *__restrict__ a, double *__restrict__ b,
                 double *__restrict__ c);
void mmtu_par(int n, double *__restrict__ a, double *__restrict__ b,
                double *__restrict__ c);

double c[N][N], b[N][N], a[N][N], cc[N][N];

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
  
  for(i=0;i<N;i++)
   for(j=0;j<N;j++)
   { a[i][j] = 1.1*(2*i+j);
     b[i][j] = 1.2*(i+2*j);
   }
  
  printf("Reference sequential code performance in GFLOPS");
  mint_seq = 1e9; maxt_seq = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   for(i=0;i<N;i++) for(j=0;j<N;j++) c[i][j] = 0;
   tstart = omp_get_wtime();
   mmtu_seq(N, &a[0][0], &b[0][0], &c[0][0]);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint_seq) mint_seq=telapsed;
   if (telapsed > maxt_seq) maxt_seq=telapsed;
  }
  printf(" Min: %.2f; Max: %.2f\n",2.0e-9*N*(N+1)*(N+2)/6/maxt_seq,2.0e-9*N*(N+1)*(N+2)/6/mint_seq);
  
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
    for(i=0;i<N;i++) for(j=0;j<N;j++) cc[i][j] = 0;
    tstart = omp_get_wtime();
    mmtu_par(N, &a[0][0], &b[0][0], &cc[0][0]);
    telapsed = omp_get_wtime()-tstart;
    if (telapsed < mint_par[nt]) mint_par[nt]=telapsed;
    if (telapsed > maxt_par[nt]) maxt_par[nt]=telapsed;
   }
   compare(N,c,cc,nthreads[nt]);
  }
  printf("Performance (Best & Worst) of parallelized version: GFLOPS on ");
  for (nt=0;nt<num_cases-1;nt++) printf("%d/",nthreads[nt]);
  printf("%d threads\n",nthreads[num_cases-1]);
  printf("Best Performance (GFLOPS): ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",2.0e-9*N*(N+1)*(N+2)/6/mint_par[nt]);
  printf("\n");
  printf("Worst Performance (GFLOPS): ");
  for (nt=0;nt<num_cases;nt++) printf("%.2f ",2.0e-9*N*(N+1)*(N+2)/6/maxt_par[nt]);
  printf("\n");
}

void compare(int n, double wref[][n], double w[][n], int numt)
{
  double maxdiff,this_diff;
  int numdiffs;
  int i,j;
  numdiffs = 0;
  maxdiff = 0;
  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      {
        this_diff = wref[i][j]-w[i][j];
        if (this_diff < 0) this_diff = -1.0*this_diff;
        if (this_diff>threshold)
          { numdiffs++;
	    printf("<%d,%d>,%f,%f",i,j,wref[i][j],w[i][j]);
            if (this_diff > maxdiff) maxdiff=this_diff;
          }
      }
  if (numdiffs > 0)
  { printf("Error when executing on %d threads; %d Differences found over threshold %f; Max Diff = %f\n",
           numt,numdiffs,threshold,maxdiff);
    printf("Exiting\n"); exit;
  
  }
}
