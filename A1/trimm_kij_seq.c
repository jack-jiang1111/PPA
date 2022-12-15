void trimm_seq(int n, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) {
int i, j, k;

for (k=0;k<n;k++)
 for (i=k;i<n;i++)
  for (j=0;j<=k;j++)
// c[i][j] = c[i][j] + a[i][k]*b[k][j];
   c[i*n+j]=c[i*n+j]+a[i*n+k]*b[k*n+j];
}
