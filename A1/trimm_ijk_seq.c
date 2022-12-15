void trimm_seq(int n, float *__restrict__ a, float *__restrict__ b,
                 float *__restrict__ c) {
int i, j, k;

for (i = 0; i < n; i++)
 for (j = 0; j <= i; j++)
  for (k = j; k <= i; k++)
// c[i][j] = c[i][j] + a[i][k]*b[k][j];
   c[i*n+j]=c[i*n+j]+a[i*n+k]*b[k*n+j];
}
