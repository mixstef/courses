#include <stdio.h>
#include <stdlib.h>

// Compile with: nvcc one-addition.cu -o one-addition
// nvcc takes care of all includes/libraries.

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


// the kernel function - must return void
__global__ void add(int a,int b,int *c) {
  
  *c = a+b;  
}


int main() {

  int c;	// host's 'c' variable
  int *dev_c;	// ptr to device's 'c' variable

  // allocate space for 'c' on device's memory
  HANDLE_ERROR(cudaMalloc((void **)&dev_c,sizeof(int)));

  // call the kernel on device, 1 block/1 thread
  // this call is asynchronous - host continues execution
  add<<<1,1>>>(2,7,dev_c);

  // transfer device's 'c' into host's 'c' - synchronous call, waits until kernel is done
  HANDLE_ERROR(cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost));

  // free memory of device's c
  HANDLE_ERROR(cudaFree(dev_c));

  // print host's 'c'
  printf("2+7=%d\n",c);

  // a catchall msg here, will catch kernel launch failures, too!
  printf( "Last error msg is: %s\n", cudaGetErrorString( cudaGetLastError() ));

  return 0;
}

