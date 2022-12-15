// For compiling, use "nvcc -O3"; upon logging on to a CHPC node, "module load cuda" is needed to load "nvcc"

#include <stdio.h>

void checkCUDAError(const char *msg);

#include <stdio.h>

#define RADIUS        3
#define BLOCK_SIZE    256
#define NUM_ELEMENTS  (4097*257)
#define FIXME1 32
#define FIXME2 32

// The FIXMEs indicate where code must replace the FIXMEs.
// cx
// The number of input elements is N+2*RADIUS, IN[0:N+2*RADIUS-1]
// Each element of out holds the sum of a set of 2*RADIUS+1 contiguous elements from in
// The sum of contents in in[0:2*RADIUS] is placed in out[0], 
// sum of elements in in[1:2*RADIUS+1] is placed in out[1], etc.

__global__ void stencil_1d(int *in, int *out, int N) 
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  //out[idx]=0;
  
  if(idx<N){
      for(int r=-RADIUS;r<=RADIUS;r++)
        out[idx] += in[RADIUS+idx+r];
      //printf("%d\n",out[idx]);
  }
}

int main()
{
  int i,r;
  int h_in[NUM_ELEMENTS + 2 * RADIUS], h_out[NUM_ELEMENTS], h_ref[NUM_ELEMENTS];
  int *d_in, *d_out;

  // Initialize host data
  for(i = 0; i < (NUM_ELEMENTS + 2*RADIUS); i++ )
    h_in[i] = i; 
  for(i = 0; i < NUM_ELEMENTS; i++)
    h_ref[i] = 0;

  for(i = 0; i < NUM_ELEMENTS; i++)
   for(r = -RADIUS; r <= RADIUS; r++)
    h_ref[i] += h_in[RADIUS+i+r];

  // Allocate space on the device
  cudaMalloc( &d_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int));
  cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int));
  checkCUDAError("cudaMalloc");

  // Copy input data to device
  cudaMemcpy( d_in, h_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy");

  // Fix the FIXME's
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(ceil(NUM_ELEMENTS/float(BLOCK_SIZE)));

  stencil_1d<<< dimGrid,dimBlock >>> (d_in, d_out,NUM_ELEMENTS);
  checkCUDAError("Kernel Launch Error:");

  cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMalloc");

  for( i = 0; i < NUM_ELEMENTS; ++i )
    if (h_ref[i] != h_out[i])
    {
      printf("ERROR: Mismatch at index %d: expected %d but found %d\n",i,h_ref[i], h_out[i]);
      break;
    }

    if (i== NUM_ELEMENTS) printf("SUCCESS!\n");

  // Free out memory
  cudaFree(d_in);
  cudaFree(d_out);

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

