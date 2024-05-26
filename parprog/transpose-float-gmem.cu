// Code example to transpose a NxN matrix of floats, GPU version, global memory only.
// Uses blocks of 256 threads, arranged in 32x8 (2D), as many required to cover NxN size.
// Each block transposes a 32x32 tile.
// compile with:  nvcc transpose-float-gmem.cu -o transpose-float-gmem -DN=4000



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


#define TILESIZE 32
#define BLOCKSIZE ((N+TILESIZE-1)/TILESIZE)


// the kernel function
__global__ void transposeTile(float *a,float *b) {

  // compute x,y position of first element in matrix this thread is going to work on
  int x = blockIdx.x * TILESIZE + threadIdx.x;
  int y = blockIdx.y * TILESIZE + threadIdx.y;
  
  // this thread will work on 32/8 = 4 elements, veritcal step is blockDim.y (=8) 
  int step = blockDim.y;
  
  for(int i=0;i<TILESIZE;i+=step) {
    // b(y+i,x) = a(x,y+i)
    if (x<N && (y+i)<N) {
      int rd_ix = (y+i)*N+x;
      int wr_ix = x*N+y+i;
      b[wr_ix] = a[rd_ix];
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

  // call the kernel on device
  dim3 blocks(BLOCKSIZE,BLOCKSIZE,1);
  dim3 threads(TILESIZE,8,1);
  transposeTile<<<blocks,threads>>>(dev_a,dev_b);
  
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

