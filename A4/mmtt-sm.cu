// For compiling, use "nvcc -O3"; upon logging on to a CHPC node, "module load cuda" is needed to load "nvcc"

#include <stdio.h>
#include <time.h>
#define threshold 0.0000001

void checkCUDAError(const char *msg);

// Dsize = 2048
const int DSIZE = 2048;
const int BLOCK_SIZE = 32;
cudaEvent_t start, stop;
float tstart, elapsedTime;

// matrix multiply kernel: C = A * B
__global__ void mmul(const double *A, const double *B, double *C, int ds) {
// Enter GPU kernel code body
  int bx = blockIdx.x;    
  int by = blockIdx.y;
  int tx = threadIdx.x; 
  int ty = threadIdx.y;
  // Starting index of A and B for the thread block
  int aBegin = BLOCK_SIZE * by;
  int bBegin = ds * BLOCK_SIZE * bx;
  //printf("aBegin: %d, bBegin: %d\n", aBegin,bBegin);
  int c = ds * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  double temp = 0;
  
  
  // Declaration of shared memory buffers 
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
  // Starting index of A and B for the thread
  int aInd = aBegin + ds * ty + tx;
  int bInd = bBegin + ds * ty + tx;

  
  for (int kt = 0; kt < ds; kt +=BLOCK_SIZE) {
    As[tx][ty] = A[aInd]; 
    Bs[tx][ty] = B[bInd];
    //printf("a , b:  %d %d\n",aInd,bInd);
    __syncthreads();
    // h_Cref[i][j] += h_A[k][i]*h_B[j][k];
    for (int k = 0; k < BLOCK_SIZE; ++k){
      //printf("%d %d A: %f B: %f %d\n",aInd,bInd,As[ty][k] , Bs[k][tx],ds * BLOCK_SIZE * by + BLOCK_SIZE * bx+ds*ty+tx);
      //printf("Val:  %f\n",ds * BLOCK_SIZE * by + BLOCK_SIZE * bx+ds*ty+tx, As[k][ty] * Bs[tx][k]);
      temp += As[ty][k] * Bs[k][tx];
    } 
    __syncthreads();
    aInd += BLOCK_SIZE * ds; 
    bInd += BLOCK_SIZE;
    C[c+ds*ty+tx]=temp;
    //int c = ds * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    //C[c+ds*ty+tx]=Csub;
  }
}

/*
notch309
03:00.0 VGA compatible controller: Matrox Electronics Systems Ltd. Integrated Matrox G200eW3 Graphics Controller (rev 04)
Matrix size: 2048
<BX=8,BY=8>: Trial 0: GFLOPS: 76.25
<BX=8,BY=8>: Trial 1: GFLOPS: 98.04
<BX=8,BY=8>: Trial 2: GFLOPS: 97.77
<BX=8,BY=8>: Trial 3: GFLOPS: 97.77
<BX=8,BY=8>: Trial 4: GFLOPS: 97.72

Matrix size: 2048
<BX=16,BY=16>: Trial 0: GFLOPS: 80.19
<BX=16,BY=16>: Trial 1: GFLOPS: 106.68
<BX=16,BY=16>: Trial 2: GFLOPS: 106.71
<BX=16,BY=16>: Trial 3: GFLOPS: 106.69
<BX=16,BY=16>: Trial 4: GFLOPS: 106.69

Matrix size: 2048
<BX=32,BY=32>: Trial 0: GFLOPS: 76.28
<BX=32,BY=32>: Trial 1: GFLOPS: 97.25
<BX=32,BY=32>: Trial 2: GFLOPS: 97.31
<BX=32,BY=32>: Trial 3: GFLOPS: 97.30
<BX=32,BY=32>: Trial 4: GFLOPS: 97.30
*/
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
  for (j=0;j<DSIZE;j++)
   for (k=0;k<DSIZE;k++)
    {
 // h_Cref[i][j] += h_A[k][i]*h_B[j][k];
    //printf("A: %f B: %f\n", h_A[k*DSIZE+i],h_B[j*DSIZE+k]);
     //printf("Ref: %d %f\n",i*DSIZE+j, h_A[k*DSIZE+i]*h_B[j*DSIZE+k]);
     h_Cref[i*DSIZE+j] += h_A[k*DSIZE+i]*h_B[j*DSIZE+k];
    }
 
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
  int Bx = BLOCK_SIZE, By = BLOCK_SIZE;
  printf("Matrix size: %d\n", DSIZE);
  
  block.x = Bx;
  block.y = By;
  grid.x = DSIZE/block.x;
  grid.y = DSIZE/block.y;

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

