// For compiling, use "nvcc -O3"; upon logging on to a CHPC node, "module load cuda" is needed to load "nvcc"

#include <stdio.h>
#include <time.h>
#define threshold 0.0000001
#define FIXME1 1
#define FIXME2 2
#define FIXME3 3
#define FIXME4 4

void checkCUDAError(const char *msg);

const int DSIZE = 1024;
cudaEvent_t start, stop;
float tstart, elapsedTime;

// matrix multiply kernel: C = A * B
__global__ void mmul(const double *A, const double *B, double *C, int ds) {
// Enter GPU kernel code body

  int idy = blockDim.x*blockIdx.x+threadIdx.x; // create thread x index
  int idx = blockDim.y*blockIdx.y+threadIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)){
    double temp = 0;
    for (int k = 0; k < ds; k++)
      temp += A[k*ds+idx] * B[idy*ds+k];   // dot product of row and column
    C[idx*ds+idy] = temp;
  }
}

int main(){

  double *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
  int i,j,k;

  h_A = new double[DSIZE*DSIZE];
  h_B = new double[DSIZE*DSIZE];
  h_C = new double[DSIZE*DSIZE];
  h_Cref = new double[DSIZE*DSIZE];
  for (i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = i-1;
    h_B[i] = i+1;
    h_C[i] = 0;
    h_Cref[i] = 0;}

  for (i=0;i<DSIZE;i++)
   for (k=0;k<DSIZE;k++)
    for (j=0;j<DSIZE;j++)
  // h_Cref[i][j] += h_A[k][i]*h_B[j][k];
     h_Cref[i*DSIZE+j] += h_A[k*DSIZE+i]*h_B[j*DSIZE+k];
 // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(double));
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(double));
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(double));
  checkCUDAError("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(double), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D failure");

  dim3 block(1,1);  
  dim3 grid(1,1);
  int Bx, By;
  printf("Matrix size: %d\n", DSIZE);
  while(1)
 {
  printf("Specify TB-size-x,TB-size-y: ");
  scanf("%d %d", &Bx,&By);
  if ((Bx==0) or (By==0)) break;
  block.x = Bx;
  block.y = By;
  grid.x = ceil(DSIZE/float(Bx));
  grid.y = ceil(DSIZE/float(By));

  for(int trial=0;trial<5;trial++)
  {
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);
   // Launch kernel
   mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
   checkCUDAError("kernel launch");
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsedTime, start,stop);
   cudaDeviceSynchronize();
   // Copy results back to host
   cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(double), cudaMemcpyDeviceToHost);
   checkCUDAError("cudaMemcpy D2H");
   for (int i = 0; i < DSIZE*DSIZE; i++) if (fabs((h_C[i] - h_Cref[i])/h_Cref[i])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", i, h_C[i], h_Cref[i]); return -1;}
   printf("<BX=%d,BY=%d>: Trial %d: GFLOPS: %.2f\n",Bx,By,trial,2.0e-6*DSIZE*DSIZE*DSIZE/elapsedTime);
  }
 }
  return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

