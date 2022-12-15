void trimm_par(int n, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) {
   
   
   int i, j, k;
   #pragma omp parallel
   {
      #pragma omp for
      for (i=0;i<n;i++){
         for (k=0;k<=i;k++){
            for (j=0;j<=k;j++){
               //c[i][j] = c[i][j] + a[i][k]*b[k][j];
               c[i*n+j]=c[i*n+j]+a[i*n+k]*b[k*n+j];
            }
         }    
      }    
   }
}
   /*
   for (k=0;k<n;k++){
      for (i=k;i<n;i++){
         int rem = (k+1)%4;
         
         for (j=0;j<rem;j++){
            c[i*n+j]=c[i*n+j]+a[i*n+k]*b[k*n+j];
         }

         #pragma omp parallel for private(j)
         for (j=rem;j<=k;j+=4){
            // c[i][j] = c[i][j] + a[i][k]*b[k][j];
            c[i*n+j]  =c[i*n+j]  +a[i*n+k]*b[k*n+j];
            c[i*n+j+1]=c[i*n+j+1]+a[i*n+k]*b[k*n+j+1];
            c[i*n+j+2]=c[i*n+j+2]+a[i*n+k]*b[k*n+j+2];
            c[i*n+j+3]=c[i*n+j+3]+a[i*n+k]*b[k*n+j+3];
         } 
      }   
   }
   */

