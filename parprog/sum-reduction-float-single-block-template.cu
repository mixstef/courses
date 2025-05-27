// Sum reduction of an array of N floats in GPU.
// This is a template version with 256 threads/1 block and shared memory usage.

// Compile with: nvcc sum-reduction-float-single-block-template.cu -o sum-reduction-float-single-block-template -DN=10000000


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
   
  int start = threadIdx.x;
  int step = blockDim.x; 
  
  // reduce my part of work
  float sum = 0.0; 
  for (int i=start;i<N;i+=step) {
    sum += a[i];
  }
  
  // store my partial result in shared memory
  buffer[start] = sum;
  
  // synchronize all threads of block
  __syncthreads();
    
  // thread 0 of block computes and writes final result to global memory
  if (start==0) {
    float sum = 0.0;
    for (int i=0;i<THREADS;i++) {
      sum += buffer[i];
    }
    *psums = sum; 
  }
}


int main() {
  float *a,sum;		// host space
  float *dev_a,*dev_sum;	// device space
  
  // allocate host input array
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  
  
  // init array to random int values from 0 to 2 (to avoid float truncation errors)
  for (int i=0;i<N;i++) {
    a[i] = rand()%3;
  }

  // allocate space on device's memory
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_sum,sizeof(float)));

  // transfer host input array to device
  HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));

  // call the kernel on device, 1 block/THREADS thread
  sumReduction<<<1,THREADS>>>(dev_a,dev_sum);

  // transfer device's sum into host's sum
  HANDLE_ERROR(cudaMemcpy(&sum,dev_sum,sizeof(float),cudaMemcpyDeviceToHost));

  // free memory of device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_sum));
  
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
