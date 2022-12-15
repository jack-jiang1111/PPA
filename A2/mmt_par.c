void mmt_par(int n, double *__restrict__ a, double *__restrict__ b,
                 double *__restrict__ c) {
int i, j, k;

   #pragma omp parallel private(i,j,k) 
   {
      
      for(k=0;k<n;k++){
         #pragma omp for
         for(i=0;i<n;i++){
            int rem = n%4;
            for (j=0;j<rem;j++){
               c[i*n+j]=c[i*n+j]+a[k*n+j]*b[k*n+i];
            }
            for (j=rem;j<n;j+=4){
               c[i*n+j]=c[i*n+j]+a[k*n+j]*b[k*n+i];
               c[i*n+j+1]=c[i*n+j+1]+a[k*n+j+1]*b[k*n+i];
               c[i*n+j+2]=c[i*n+j+2]+a[k*n+j+2]*b[k*n+i];
               c[i*n+j+3]=c[i*n+j+3]+a[k*n+j+3]*b[k*n+i];
            }
         }
      }     
   }
}
/*

*/
