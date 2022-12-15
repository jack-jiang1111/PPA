#include <omp.h>
void mv2_par(int n, double *__restrict__ m, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z)
{
  int i,j,i_,j_;
  #pragma omp parallel private(i,j,i_,j_)
  {
      
      for (i = 0; i < n; i++){
        #pragma omp for
        for (j = 0; j < n; j++) {
          //y[j] = y[j] + m[i][j] * x[i];
          y[j] = y[j] + m[i*n+j] * x[i];
          //z[j] = z[j] + m[j*n+i] * x[i];
        }
      }

      #pragma omp for
      for (j_ = 0; j_ < n; j_++) {
        for (i_ = 0; i_ < n; i_++){
          z[j_] = z[j_] + m[j_*n+i_] * x[i_];
        }
      }
  }
}

