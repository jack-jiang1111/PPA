void mmt_seq(int n, double *__restrict__ a, double *__restrict__ b,
                 double *__restrict__ c) {
int i, j, k;

for(i=0;i<n;i++)
 for(j=0;j<n;j++)
  for(k=0;k<n;k++)
//    c[i][j] = c[i][j] + a[k][j]*b[k][i];
   c[i*n+j]=c[i*n+j]+a[k*n+j]*b[k*n+i];
}
