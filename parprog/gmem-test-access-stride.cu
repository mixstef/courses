// Sample program to test global memory access, 256 threads/multiple blocks.
// This version uses a parametric stride to array indexing.
// Compile with: nvcc gmem-test-access-stride.cu -o gmem-test-access-stride -DN=25600000 -DSTRIDE=1

// NOTE: array size is rounded up to 256 (= grid size). Stride surplus is added.
// All threads are working, each one handling exactly one array element.

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


// the kernel function
__global__ void test_incr(float *a) {
 
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * STRIDE;
 
  a[i] += 1.1;	// sample operation

}


int main() {

  int threads = 256;
  int blocks = (N + threads - 1)/threads;
  int ceil_n = threads*blocks*STRIDE;
  
  printf("Array elements: %d (%d threads x %d blocks x %d stride)\n",ceil_n,threads,blocks,STRIDE);
  
  // allocate space on device's memory
  float *dev_a;	// device's space ptr
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,ceil_n*sizeof(float)));
  
  // Init device memory bytes to sample value (0)
  HANDLE_ERROR(cudaMemset(dev_a,0,ceil_n*sizeof(float)));

  // create CUDA events for current device
  cudaEvent_t start,stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  // insert start event into default stream 0
  HANDLE_ERROR(cudaEventRecord(start,0));
     
  // call the kernel on device, "blocks" blocks/256 threads
  test_incr<<<blocks,threads>>>(dev_a);

  // insert stop event into default stream 0
  HANDLE_ERROR(cudaEventRecord(stop,0));

  // block host until stop event has been reached on device
  HANDLE_ERROR(cudaEventSynchronize(stop));

  // measure time diff between start-stop
  float millisecs;
  HANDLE_ERROR(cudaEventElapsedTime(&millisecs,start,stop));
    
  // free memory on device
  HANDLE_ERROR(cudaFree(dev_a));
  
  // print kernel's exec time (NOTE: this includes kernel launch delay)
  printf("Kernel execution time: %f ms\n",millisecs); 
 
  // a catchall msg here, will catch kernel launch failures, too!
  printf("Last CUDA error msg is: %s\n", cudaGetErrorString( cudaGetLastError() ));
  
  return 0;
}
