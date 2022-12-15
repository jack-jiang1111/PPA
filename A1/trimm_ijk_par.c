
#include <stdio.h>
void trimm_par(int n, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) {
int i, j, k;
  #pragma omp parallel
  {
    #pragma omp for
    for (i=0;i<n;i++){
         //#pragma omp for private(i)
      for (k=0;k<=i;k++){
        int rem = (k+1)%4;
         
        for (j=0;j<rem;j++){
          c[i*n+j]=c[i*n+j]+a[i*n+k]*b[k*n+j];
        }
        for (j=rem;j<=k;j+=4){
          // c[i][j] = c[i][j] + a[i][k]*b[k][j];
          c[i*n+j]+=a[i*n+k]*b[k*n+j];
          c[i*n+j+1]+=a[i*n+k]*b[k*n+j+1];
          c[i*n+j+2]+=a[i*n+k]*b[k*n+j+2];
          c[i*n+j+3]+=a[i*n+k]*b[k*n+j+3];
        }  
      }
    }    
  }
}


/*
// unrolling inner loop
        int rem = (i+1-j)%4;
         
        for (k=j;k<rem;k++){
          c[i*n+j]=c[i*n+j]+a[i*n+k]*b[k*n+j];
        }

        for (k=rem;k<=i;k+=4){
          // c[i][j] = c[i][j] + a[i][k]*b[k][j];
          c[i*n+j]+=a[i*n+k]*b[k*n+j];z
          c[i*n+j]+=a[i*n+k+1]*b[(k+1)*n+j];
          c[i*n+j]+=a[i*n+k+2]*b[(k+2)*n+j];
          c[i*n+j]=a[i*n+k+3]*b[(k+3)*n+j];
        }  
*/