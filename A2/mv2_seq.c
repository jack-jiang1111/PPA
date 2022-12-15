void mv2_seq(int n, double *__restrict__ m, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z)
{
  int i, j;
  for (i = 0; i < n; i++)
   for (j = 0; j < n; j++) {
//    y[j] = y[j] + m[i][j] * x[i];
//    z[j] = z[j] + m[j][i] * x[i];
//
      y[j] = y[j] + m[i*n+j] * x[i];
      z[j] = z[j] + m[j*n+i] * x[i];
   }
}
