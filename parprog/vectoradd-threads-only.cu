// Example of CUDA parallel float vector addition using 1 block/N threads
// Compile with: nvcc vectoradd-threads-only.cu -o vectoradd-threads-only

#include <stdio.h>
#include <stdlib.h>


#define N 100


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
 
  int i = threadIdx.x; 
  
  c[i] = a[i]+b[i];
}


int main() {
  float *a,*b,*c;		// host's space ptrs
  float *dev_a,*dev_b,*dev_c;	// device's space ptrs

  // allocate space on host's memory
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) exit(1);
  b = (float *)malloc(N*sizeof(float));
  if (b==NULL) { free(a); exit(1); }
  c = (float *)malloc(N*sizeof(float));
  if (c==NULL) { free(a); free(b); exit(1); }

  // init host arrays
  for (int i=0;i<N;i++) {
    a[i] = (float)(-i);
    b[i] = (float)(2*i);
  }

  // allocate space on device's memory
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c,N*sizeof(float)));

  // transfer host arrays to device
  HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice));

  // call the kernel on device, 1 block/N threads
  vector_add<<<1,N>>>(dev_a,dev_b,dev_c);

  // transfer device's 'c' into host's 'c' array
  HANDLE_ERROR(cudaMemcpy(c,dev_c,N*sizeof(float),cudaMemcpyDeviceToHost));

  // free memory of device
  HANDLE_ERROR(cudaFree(dev_c));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_a));

  // check results
  for (int i=0;i<N;i++) {
    if (c[i]!=a[i]+b[i]) {
      printf("%f %f Result error!\n",c[i],a[i]+b[i]);
      break;
    }
  }

  // free memory on host
  free(c); free(b); free(a);
  
  // a catchall msg here, will catch kernel launch failures, too!
  printf("Last CUDA error msg is: %s\n", cudaGetErrorString( cudaGetLastError() ));

  return 0;
}

