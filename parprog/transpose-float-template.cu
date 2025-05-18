// Example to transpose a NxN matrix of floats in GPU, global memory only.

// NOTE: this is only a template, using a single thread to transpose
// the entire matrix, should be transformed to something more useful! 

// Compile with:  nvcc transpose-float-template.cu -o transpose-float-template -DN=4000



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
__global__ void transpose(float *a,float *b) {

  for (int i=0;i<N;i++) { // for every row
    for (int j=0;j<N;j++) { // for every column
      b[j*N+i] = a[i*N+j];// b(j,i) = a(i,j)
    }
  }
}


int main() {
float *a,*b;
float *dev_a,*dev_b;


  a = (float *)malloc(N*N*sizeof(float)); 
  if (a==NULL) {
    printf("alloc error!\n");
    exit(1);
  }

  b = (float *)malloc(N*N*sizeof(float)); 
  if (b==NULL) {
    printf("alloc error!\n");
    free(a);
    exit(1);
  }

  // init input array
  for (int i=0;i<N*N;i++) {
     a[i] = (float)rand()/RAND_MAX;
  } 

  // allocate space on device's memory
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b,N*N*sizeof(float)));

  // transfer host input array to device
  HANDLE_ERROR(cudaMemcpy(dev_a,a,N*N*sizeof(float),cudaMemcpyHostToDevice));

  // launch the kernel on device
  transpose<<<1,1>>>(dev_a,dev_b);
  
  // transfer device's output into host's output array
  HANDLE_ERROR(cudaMemcpy(b,dev_b,N*N*sizeof(float),cudaMemcpyDeviceToHost));

  // free memory of device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));

  // check operation
  int err = 0;
  for (int i=0;i<N && err!=1;i++) {
    for (int j=0;j<N;j++) {
      if (b[j*N+i] != a[i*N+j]) {
        printf("Error!\n");
        err = 1;
        break;
      }
    }
  }
  
  if (err==0) printf("Success!\n");


  free(b);
  free(a);

   // a catchall msg here, will catch kernel launch failures, too!
  printf("Last CUDA error msg is: %s\n", cudaGetErrorString( cudaGetLastError() ));
 
  return 0;
}

