// Sum reduction of an array of N floats in GPU.
// This is a template version with 256 threads/fixed number of blocks + grid striding and shared memory usage.

// Compile with: nvcc sum-reduction-float-grid-striding.cu -o sum-reduction-float-grid-striding -DN=10000000


#include <stdio.h>
#include <stdlib.h>


// helper function and macro
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define THREADS 256


// the kernel function
__global__ void sumReduction(float *a,float *psums) {
  
  __shared__ float buffer[THREADS];
   
  int start = blockIdx.x*blockDim.x + threadIdx.x;
  int step = blockDim.x*gridDim.x; 
  
  // reduce my part of work
  float sum = 0.0; 
  for (int i=start;i<N;i+=step) {
    sum += a[i];
  }
  
  // store my partial result in shared memory
  buffer[threadIdx.x] = sum;
  
  // synchronize all threads of block
  __syncthreads();
    
  // thread 0 of block computes and writes final result to global memory
  if (threadIdx.x==0) {
    float sum = 0.0;
    for (int i=0;i<THREADS;i++) {
      sum += buffer[i];
    }
    psums[blockIdx.x] = sum; 
  }
}


int main() {
  float *a,*sums;		// host space
  float *dev_a,*dev_sums;	// device space
  
  // compute number of blocks in grid (32 x #SMs)
  int devId;
  HANDLE_ERROR(cudaGetDevice(&devId));
  int numSM;
  HANDLE_ERROR(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, devId));
  int blocks = numSM*32;	// as a multiple of SMs in GPU


  // allocate host input array
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  
  // allocate host partial results array (size equal to num of blocks)
  sums = (float *)malloc(blocks*sizeof(float));
  if (sums==NULL) {
    printf("Allocation failed!\n");
    free(a); exit(1);
  }	  
  
  // init array to random int values from 0 to 2 (to avoid float truncation errors)
  for (int i=0;i<N;i++) {
    a[i] = rand()%3;
  }

  // allocate space on device's memory
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_sums,blocks*sizeof(float)));

  // transfer host input array to device
  HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));

  // call the kernel on device
  printf("Launching kernel with %d blocks, of %d threads each.\n",blocks,THREADS);
  sumReduction<<<blocks,THREADS>>>(dev_a,dev_sums);

  // transfer device's partial sums into host's sums
  HANDLE_ERROR(cudaMemcpy(sums,dev_sums,blocks*sizeof(float),cudaMemcpyDeviceToHost));

  // free memory of device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_sums));
  
  // combine partial sums on CPU
  float sum = 0.0;
  for (int i=0;i<blocks;i++) {
    sum += sums[i];
  }
    
  // check result, computed on cpu
  float check = 0.0;
  for (int i=0;i<N;i++) {
    check += a[i];
  }
  if (check!=sum) {
    printf("Error! found %f instead of %f\n",sum,check);
  }
  else {
    printf("Success, result = %f\n",sum);
  }  
  free(a);
 
   // a catchall msg here, will catch kernel launch failures, too!
  printf("Last CUDA error msg is: %s\n", cudaGetErrorString( cudaGetLastError() ));
 
  return 0;
}
