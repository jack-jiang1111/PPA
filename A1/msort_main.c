// Use "gcc -O3 -fopenmp msort_main.c msort_seq.c msort_par.c" to compile

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#ifndef N
#define N (10000001)
#endif
#ifndef NTrials
#define NTrials (3)
#endif

void msort_seq(int a[],int b[],int lo, int hi);
void msort_par(int a[],int b[],int lo, int hi);
void Test_Sorted(int a[],int lo,int hi);


int A[N],B[N];


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
  
  printf("List Size = %d\n",N);

  mint_seq = 1e9; maxt_seq = 0;
  for(trial=0;trial<NTrials;trial++)
  {
   for(i=0;i<N;i++)  A[i] = N-i;
   tstart = omp_get_wtime();
   msort_seq(A,B, 0, N-1);
   telapsed = omp_get_wtime()-tstart;
   if (telapsed < mint_seq) mint_seq=telapsed;
   if (telapsed > maxt_seq) maxt_seq=telapsed;
  }
  printf("Min/Max sequential Sort Rate: %.1f/%.1f Mega-Elements/Second\n", 1.0*N/maxt_seq/1000000, 1.0*N/mint_seq/1000000);
  
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
    for(i=0;i<N;i++)  A[i] = N-i;
    tstart = omp_get_wtime();
    msort_par(A,B, 0, N-1);
    telapsed = omp_get_wtime()-tstart;
    if (telapsed < mint_par[nt]) mint_par[nt]=telapsed;
    if (telapsed > maxt_par[nt]) maxt_par[nt]=telapsed;
   }
   Test_Sorted(A,0,N-1);
  }
  printf("Best & Worst Performance of parallelized version: Mega-Elts/second || Speedup on ");
  for (nt=0;nt<num_cases-1;nt++) printf("%d/",nthreads[nt]);
  printf("%d threads\n",nthreads[num_cases-1]);
  printf("Best Performance (Mega-Elts/second || Speedup): ");
  for (nt=0;nt<num_cases;nt++) printf("%.1f ",1.0e-6*N/mint_par[nt]);
  printf(" || ");
  for (nt=0;nt<num_cases;nt++) printf("%.1f ",mint_seq/mint_par[nt]);
  printf("\n");
  printf("Worst Performance (Mega-Elts/second || Speedup): ");
  for (nt=0;nt<num_cases;nt++) printf("%.1f ",1.0e-6*N/maxt_par[nt]);
  printf(" || ");
  for (nt=0;nt<num_cases;nt++) printf("%.1f ",mint_seq/maxt_par[nt]);
  printf("\n");
}

void Test_Sorted(a,lo,hi)
int a[],lo,hi;
{ int i, notsorted;
  notsorted = 0;
  for(i=lo+1;i<=hi;i++)
   if(a[i-1]>a[i]) printf("Sorting error: A[%d]=%d; A[%d]=%d\n",i-1,a[i-1],i,a[i]);
//   if(a[i-1]>a[i]) notsorted++;
  if (notsorted>0) printf("Error: sequence is not sorted in increasing order; # flips =%d\n",notsorted);
}

