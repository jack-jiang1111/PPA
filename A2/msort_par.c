#include "omp.h"
void merge(int a[],int b[],int lo,int mid,int hi);
void msort_seq(int a[],int b[],int lo, int hi);


void msort_par(int a[],int b[],int lo, int hi)
{
  
  int temp,mid;
  if (lo < hi){
    if (hi == lo+1){ 
      if (a[hi]<a[lo]) {
        temp=a[hi];a[hi]=a[lo];a[lo]=temp;
      }
    }
    else{
      // seperate region into 8 picies
      // part a to part i
      int pa = lo;
      int pi = hi;
      int pe = (lo+hi)/2;
      int pc = (pa+pe)/2;
      int pg = (pe+pi)/2;
      int pb = (pa+pc)/2;
      int pd = (pc+pe)/2;
      int pf = (pe+pg)/2;
      int ph = (pg+pi)/2;
      #pragma omp parallel
      {
        #pragma omp sections 
        {
          #pragma omp section
            msort_seq(a,b,pa,pb);
          #pragma omp section
            msort_seq(a,b,pb+1,pc);
          #pragma omp section
            msort_seq(a,b,pc+1,pd);
          #pragma omp section
            msort_seq(a,b,pd+1,pe);
          #pragma omp section
            msort_seq(a,b,pe+1,pf);
          #pragma omp section
            msort_seq(a,b,pf+1,pg);
          #pragma omp section
            msort_seq(a,b,pg+1,ph);
          #pragma omp section
            msort_seq(a,b,ph+1,pi);
        }
      }

      merge(a,b,pa,pb,pc);

      merge(a,b,pc+1,pd,pe);

      merge(a,b,pe+1,pf,pg);

      merge(a,b,pg+1,ph,pi);

      merge(a,b,pa,pc,pe);
  
      merge(a,b,pe+1,pg,pi);
       
      merge(a,b,pa,pe,pi);
    }
  }
}

