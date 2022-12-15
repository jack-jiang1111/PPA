#include <stdio.h>
#include <time.h>
#define threshold 0.0000001

void checkCUDAError(const char *msg);

cudaEvent_t start, stop;
float tstart, elapsedTime;

__global__ void atbt(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {
  int col = blockIdx.y*blockDim.y+threadIdx.y; // j
  int row = blockIdx.x*blockDim.x+threadIdx.x; // i
  
  if((row<Ni)&&(col<Nj)){
    double temp = 0;
    
    for(int k=0;k<Nk;k++){
      //C[i][j] += A[k][i]*B[j][k];
      temp += A[k*Ni+row]*B[col*Nk+k];
    }
    C[row*Nj+col] = temp;
  }
}
__global__ void atbtUnrollk(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {
// Initially empty; will clearly not pass correctness
  int col = blockIdx.y*blockDim.y+threadIdx.y; // j
  int row = blockIdx.x*blockDim.x+threadIdx.x; // i
  
  if((row<Ni)&&(col<Nj)){
    double temp = 0;
    int rem = Nk%2;
    for(int k=0;k<rem;k++){
      temp += A[k*Ni+row]*B[col*Nk+k];
    }
    for(int k=rem;k<Nk;k+=2){
      //C[i][j] += A[k][i]*B[k][j];
      temp += A[k*Ni+row]*B[col*Nk+k];
      temp += A[(k+1)*Ni+row]*B[col*Nk+k+1];
    }
    C[row*Nj+col] = temp;
  }
}

__global__ void atbtUnrollj(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {
// Initially empty; will clearly not pass correctness
  int col = (blockIdx.y*blockDim.y+threadIdx.y)*2; // j
  int row = blockIdx.x*blockDim.x+threadIdx.x; // i
  
  if((row<Ni)&&(col<Nj)){
    double temp = 0;
    double temp2 = 0;
    for(int k=0;k<Nk;k++){
      //C[i][j] += A[k][i]*B[k][j];
      temp += A[k*Ni+row]*B[col*Nk+k];
      temp2 += A[k*Ni+row]*B[(col+1)*Nk+k];
    }
    //printf("index %d %d \n",row,col);
    C[row*Nj+col] = temp;
    if(col+1!=Nj){ // the last index offset

    C[row*Nj+col+1] = temp2;}
    //printf("index %d %d \n",row,col);
  }
}
__global__ void atbtUnrolli(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {
// Initially empty; will clearly not pass correctness
  int col = blockIdx.y*blockDim.y+threadIdx.y; // j
  int row = blockIdx.x*blockDim.x*2+threadIdx.x; // i
  
  if((row<Ni)&&(col<Nj)){
    double temp = 0;
    double temp2 = 0;
    for(int k=0;k<Nk;k++){
      //C[i][j] += A[k][i]*B[k][j];
      temp += A[k*Ni+row]*B[col*Nk+k];
      temp2 += A[k*Ni+row+blockDim.x]*B[col*Nk+k];
    }
    C[row*Nj+col] = temp;
    C[(row+blockDim.x)*Nj+col] = temp2;
  }
}
__global__ void atbtUnrolla(const double *A, const double *B, double *C, int Ni, int Nj, int Nk) {
// Initially empty; will clearly not pass correctness
  int col = (blockIdx.y*blockDim.y+threadIdx.y)*2; // j
  int row = blockIdx.x*blockDim.x*2+threadIdx.x; // i
  
  if((row<Ni)&&(col<Nj)){
    double temp = 0;
    double temp1 = 0;
    double temp2 = 0;
    double temp3 = 0;
    int rem = Nk%2;
    for(int k=0;k<rem;k++){
      temp += A[k*Ni+row]*B[col*Nk+k];
      temp1 += A[k*Ni+row+blockDim.x]*B[col*Nk+k];
      temp2 += A[k*Ni+row]*B[(col+1)*Nk+k];
      temp3 += A[k*Ni+row+blockDim.x]*B[(col+1)*Nk+k];
    }
    for(int k=rem;k<Nk;k++){
      //C[i][j] += A[k][i]*B[k][j];
      temp += A[k*Ni+row]*B[col*Nk+k];
      temp1 += A[k*Ni+row+blockDim.x]*B[col*Nk+k];
      temp2 += A[k*Ni+row]*B[(col+1)*Nk+k];
      temp3 += A[k*Ni+row+blockDim.x]*B[(col+1)*Nk+k];
    }
    C[row*Nj+col] = temp;
    C[(row+blockDim.x)*Nj+col] = temp1;
    if(col+1!=Nj){ // the last index offset
    C[row*Nj+col+1] = temp2;
    C[(row+blockDim.x)*Nj+col+1] = temp3;}
  }
}
int main(){

  double *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
  int i,j,k,Ni,Nj,Nk;
  while(1){
    printf("Specify Matrix dimension Ni, Nj, Nk: ");
    scanf("%d %d %d", &Ni,&Nj,&Nk);
    h_A = (double *) malloc(sizeof(double)*Ni*Nk);
    h_B = (double *) malloc(sizeof(double)*Nk*Nj);
    h_C = (double *) malloc(sizeof(double)*Ni*Nj);
    h_Cref = (double *) malloc(sizeof(double)*Ni*Nj);
    for (i=0; i<Ni; i++)
    for (k=0; k<Nk; k++)
      h_A[i*Nk+k] = i*Nk+k-1;
    for (k=0; k<Nk; k++)
    for (j=0; j<Nj; j++)
      h_B[k*Nj+j] = k*Nj+j+1;
    for (i=0; i<Ni; i++)
    for (j=0; j<Nj; j++) {
      h_C[i*Nj+j] = 0;
      h_Cref[i*Nj+j] = 0;}

    for (i=0;i<Ni;i++)
    for (k=0;k<Nk;k++)
      for (j=0;j<Nj;j++)
    // h_Cref[i][j] += h_A[k][i]*h_B[j][k];
      h_Cref[i*Nj+j] += h_A[i+Ni*k]*h_B[k+Nk*j];
    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, Nk*Ni*sizeof(double));
    cudaMalloc(&d_B, Nj*Nk*sizeof(double));
    cudaMalloc(&d_C, Ni*Nj*sizeof(double));
    checkCUDAError("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, Nk*Ni*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Nj*Nk*sizeof(double), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy H2D failure");

    dim3 block(1,1);
    dim3 grid(1,1);
    int Bx, By;
    printf("Input Tbx and Tby: ");
    scanf("%d %d", &Bx,&By);
    if ((Bx<=0) || (By<=0)) {
      Bx = 1;
      By = 1;
      // prevent bad input
      printf("Invalid Bx and By\n");
    }
    char unrolling;
    printf("Unrolling any loop? Input sinle char: i,j,k, or n (no unrolling), a (all unrolling)\n");
    scanf(" %c", &unrolling);
    block.x = Bx;
    block.y = By;
    if(unrolling=='i'){
      grid.x = ceil((float)Ni/block.x/2);
      grid.y = ceil((float)Nj/block.y);
    }
    else if(unrolling=='j'){
      grid.x = ceil((float)Ni/block.x);
      grid.y = ceil((float)Nj/block.y/2);
    }
    else if(unrolling=='a'){
      grid.x = ceil((float)Ni/block.x/2);
      grid.y = ceil((float)Nj/block.y/2);
    }
    else{
      grid.x = ceil((float)Ni/block.x);
      grid.y = ceil((float)Nj/block.y);
    }
    
    printf("grid: %d %d\n",grid.x,grid.y);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int trial=0;trial<5;trial++)
    {
      cudaEventRecord(start);
      // Launch kernel
    if(unrolling=='i'){
      atbtUnrolli<<<grid, block>>>(d_A, d_B, d_C, Ni,Nj,Nk);
    }
    else if(unrolling=='j'){
      atbtUnrollj<<<grid, block>>>(d_A, d_B, d_C, Ni,Nj,Nk);
    }
    else if(unrolling=='k'){
      atbtUnrollk<<<grid, block>>>(d_A, d_B, d_C, Ni,Nj,Nk);
    }
    else if(unrolling=='a'){
      atbtUnrolla<<<grid, block>>>(d_A, d_B, d_C, Ni,Nj,Nk);
    } 
    else{
      atbt<<<grid, block>>>(d_A, d_B, d_C, Ni,Nj,Nk);
    }  
      cudaEventRecord(stop);
      checkCUDAError("kernel launch");
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsedTime, start,stop);
    //   cudaDeviceSynchronize();
      // Copy results back to host
      cudaMemcpy(h_C, d_C, Ni*Nj*sizeof(double), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy D2H");
      for (int l = 0; l < Ni*Nj; l++) if (fabs((h_C[l] - h_Cref[l])/h_Cref[l])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", l, h_C[l], h_Cref[l]); return -1;}
      printf("<Ni=%d,Nj=%d,Nk=%d>: Trial %d: GFLOPS: %.2f\n",Ni,Nj,Nk,trial,2.0e-6*Ni*Nj*Nk/elapsedTime);
    }
  }
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


