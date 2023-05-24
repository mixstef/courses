// Program that computes dot product of 2 vectors
// of length N, using shared memory for accumulation of partial products.
// This version uses numSM*32 blocks, of 256 threads each and grid-striding loop. 

// Compile with: nvcc dotproduct-shm.cu -o dotproduct-shm


#include <stdio.h>
#include <stdlib.h>


#define N 100000000

#define THREADS 256


// macro for checking result
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)


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

__global__ void dotproduct(float *a,float *b,float *c) {

  __shared__ float buf[THREADS];	// in shared memory of each block  
  
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  
  // phase 1: accumulate products by grid striding
  float temp = 0;
  for (int i=tid; i<N; i += blockDim.x * gridDim.x) {
    temp += a[i]*b[i];	// accumulate products
  }
  // store partial sum in temp buffer
  buf[threadIdx.x] = temp;
  
  // sync between phases, all threads in block
  __syncthreads();
  
  // phase 2: reduce block. Number of threads in block *must* be power of 2!!
  int stride = blockDim.x/2;
  while (stride!=0) {
    if (threadIdx.x<stride) {
      buf[threadIdx.x] += buf[threadIdx.x+stride];	// add upper half of buffer to lower half
    }
    
    __syncthreads();	// sync on one reduction pass. 
    
    stride /= 2;		// and proceed to next reduction
  }
  
  // thread 0 of block writes partial result to output
  if (threadIdx.x==0) {
    c[blockIdx.x] = buf[0];
  }
}



// main program

int main() {
int i;
float *a,*b,*c;		// host's space ptrs
float *dev_a,*dev_b,*dev_c;	// device's space ptrs

  // calculate blocks/grid as a multiple of SM number  
  int devId;
  HANDLE_ERROR(cudaGetDevice(&devId));
  int numSM;
  HANDLE_ERROR(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, devId));
  int blocks = numSM*32;	// as a multiple of SMs in GPU
    
  // allocate space on host's memory
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) exit(1);
  b = (float *)malloc(N*sizeof(float));
  if (b==NULL) { free(a); exit(1); }
  c = (float *)malloc(blocks*sizeof(float));
  if (c==NULL) { free(a); free(b); exit(1); }
 
  // allocate space on device's memory
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c,blocks*sizeof(float)));
  
  // sample init host's arrays
  for (i=0;i<N;i++) {
    a[i] = i;
    b[i] = 2*i;
  }
  
  // transfer host arrays to device
  HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice));

  // create timing events
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // mark start time
  cudaEventRecord(start,0);
  
  // call the kernel on device
  dotproduct<<<blocks,THREADS>>>(dev_a,dev_b,dev_c);

  // mark stop time
  cudaEventRecord(stop,0);
  
  // synchronize (wait "stop" mark to be actually executed)
  cudaEventSynchronize(stop);

  // compute and output stop-start time
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));
  printf("Elapsed on GPU: %f ms\n",elapsedTime);
  
  // 'free' events
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  
  // transfer device's 'c' into host's 'c' array
  HANDLE_ERROR(cudaMemcpy(c,dev_c,blocks*sizeof(float),cudaMemcpyDeviceToHost));

  // finish summation on CPU
  float sum = 0;
  for (i=0;i<blocks;i++) sum += c[i];
  
  // math check of result
  printf("Result is       : %.6g\n",sum);
  printf("Result should be: %.6g\n",2*sum_squares((float)(N-1)));
  
  // free memory of device
  HANDLE_ERROR(cudaFree(dev_c));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_a));

  // free memory on host
  free(c); free(b); free(a);
  
  // a catchall msg here, will catch kernel launch failures, too!
  printf( "Last CUDA error msg: %s\n", cudaGetErrorString( cudaGetLastError() ));
  
  return 0;
}
