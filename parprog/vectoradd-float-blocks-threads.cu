// Sample vector addition performed on GPU, 256 threads/multiple blocks.
// Compile with: nvcc vectoradd-float-blocks-threads.cu -o vectoradd-float-blocks-threads -DN=10000000

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
__global__ void vector_add(float *a,float *b,float *c) {
 
  int i = blockDim.x * blockIdx.x + threadIdx.x;
 
  if (i<N) {
    c[i] = a[i]+b[i];
  }  

}


int main() {
  float *a,*b,*c;		// host's space ptrs
  float *dev_a,*dev_b,*dev_c;	// device's space ptrs
  
  // allocate space on host's memory
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) { printf("Allocation failed!\n"); exit(1); }
  b = (float *)malloc(N*sizeof(float));
  if (b==NULL) { printf("Allocation failed!\n"); free(a); exit(1); }
  c = (float *)malloc(N*sizeof(float));
  if (c==NULL) { printf("Allocation failed!\n"); free(a); free(b); exit(1); }
 
  //initialize host arrays - cache warm-up
  for (int i=0;i<N;i++) {
    a[i]=2.0*i;
    b[i]=-i;
    c[i]=i+5.0;
  }

  // allocate space on device's memory
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c,N*sizeof(float)));

  // transfer host arrays to device
  HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice));
   
  // do artificial work
  int threads = 256;
  int blocks = (N + threads - 1)/threads;
  // call the kernel on device, "blocks" blocks/256 threads
  vector_add<<<blocks,threads>>>(dev_a,dev_b,dev_c);

  // transfer device's 'c' into host's 'c' array
  HANDLE_ERROR(cudaMemcpy(c,dev_c,N*sizeof(float),cudaMemcpyDeviceToHost));

  // free memory of device
  HANDLE_ERROR(cudaFree(dev_c));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_a));
   
  // check result - avoid loop removal by compiler
  for (int i=0;i<N;i++) {
    if (c[i]!=a[i]+b[i]) {
      printf("Error!\n");
      break;
    }
  }
   
  // free arrays
  free(a); free(b); free(c);
 
  // a catchall msg here, will catch kernel launch failures, too!
  printf("Last CUDA error msg is: %s\n", cudaGetErrorString( cudaGetLastError() ));
  
  return 0;
}
