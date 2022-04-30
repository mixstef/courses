// sum reduction of an array of floats using OpenMP SIMD construct + threads
// compile with: gcc -msse2 -Wall -O2 -fopenmp sum-float-simd-threads.c -o sum-float-simd-threads -DN=10000 -DR=100000

// NOTE1: float result has not the accuracy required by big values of N!
// NOTE2: order of additions is NOT the same as in serial reduction (may lead to different result)!
// NOTE3: N must be a multiple of 4!

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <omp.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}




int main() {
  double ts,te;
  float *a,fsum;
  
  
  // 1. allocate array
  int i = posix_memalign((void **)&a,16,N*sizeof(float));
  if (i!=0) {
    printf("Allocation failed!\n");
    exit(1);
  }	  
  
  // 2. init array to 1..N
  for (int i=0;i<N;i++) {
    a[i] = i+1;
  }


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  #pragma omp parallel
  {
    // 3. reduce array to sum (R times)
    for (int j=0;j<R;j++) {
      fsum = 0.0;
    
      #pragma omp for simd aligned(a:16) reduction(+:fsum) nowait
      for (int i=0;i<N;i++) {
        fsum += a[i];
      }
    }
  }
    
  // get ending time
  get_walltime(&te);
  
  // 4. print result (no check due to float rounding)
  printf("Result = %f\n",fsum);

  // 5. free array
  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
