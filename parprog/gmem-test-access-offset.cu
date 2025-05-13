// Sample program to test global memory access, 256 threads/multiple blocks.
// This version uses a parametric offset to array indexing.
// Compile with: nvcc gmem-test-access-offset.cu -o gmem-test-access-offset -DN=25600000 -DOFFS=0

// NOTE: array size is rounded up to 256 (= grid size). Offset surplus is added.
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
 
  int i = blockDim.x * blockIdx.x + threadIdx.x + OFFS;
 
  a[i] += 1.1;	// sample operation

}


int main() {

  int threads = 256;
  int blocks = (N + threads - 1)/threads;
  int ceil_n = threads*blocks+OFFS;
  
  printf("Array elements: %d (%d threads x %d blocks + %d for offset)\n",ceil_n,threads,blocks,OFFS);
  
  // allocate space on device's memory
  float *dev_a;	// device's space ptr
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,ceil_n*sizeof(float)));
  
  // Init device memory bytes to sample value (0)
  HANDLE_ERROR(cudaMemset(dev_a,0,ceil_n*sizeof(float)));
     
  // call the kernel on device, "blocks" blocks/256 threads
  test_incr<<<blocks,threads>>>(dev_a);

  // wait for all previous operations on device to complete
  HANDLE_ERROR(cudaDeviceSynchronize());
   
  // free memory on device
  HANDLE_ERROR(cudaFree(dev_a));
  
  // a catchall msg here, will catch kernel launch failures, too!
  printf("Last CUDA error msg is: %s\n", cudaGetErrorString( cudaGetLastError() ));
  
  return 0;
}
