// Example to transpose a NxN matrix of floats in GPU, shared memory used.

// Uses blocks of 1024 threads, arranged in 32x32 (2D) tiles, as many required to cover NxN size.
// Each block transposes a 32x32 tile in shared memory.

// Compile with:  nvcc transpose-float-smem.cu -o transpose-float-smem -DN=4000



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
#define BLOCKS ((N+TILESIZE-1)/TILESIZE)


// the kernel function
__global__ void transposeTileShm(float *a,float *b) {

  // shared memory for a 32x32 tile (2D array)
  __shared__ float buffer[TILESIZE][TILESIZE];
  // NOTE: change to buffer[TILESIZE][TILESIZE+1] to avoid bank conflicts!
  
  // // compute x,y position of input element for this thread
  int x = blockIdx.x * TILESIZE + threadIdx.x;  // input column
  int y = blockIdx.y * TILESIZE + threadIdx.y;  // input row

  // load input tile from global memory and copy into shared memory
  if (x<N && y<N) {    
    buffer[threadIdx.y][threadIdx.x] = a[y*N+x]; // buffer(thread-y,thread-x) = a(y,x)
  }
      
  // sync needed, threads use shared memory data other than their own
  __syncthreads();
  
  // compute x,y position of output element for this thread
  // NOTE: (blockIdx.y * TILESIZE,blockIdx.x * TILESIZE) is the base of transposed block.
  // threadIdx.x and threadIdx.y thread offsets are same as in input phase
  x = blockIdx.y * TILESIZE + threadIdx.x;
  y = blockIdx.x * TILESIZE + threadIdx.y;
  
  // transpose and store output element to global memory
  if (x<N && y<N) {    
    b[y*N+x] = buffer[threadIdx.x][threadIdx.y]; // b(y,x) = buffer(thread-x,thread-y)
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
  dim3 blocks(BLOCKS,BLOCKS,1);
  dim3 threads(TILESIZE,TILESIZE,1);
  transposeTileShm<<<blocks,threads>>>(dev_a,dev_b);
  
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

