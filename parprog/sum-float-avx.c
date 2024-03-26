// Sum reduction of an array of N floats, repeated R times, using AVX instructions
// Final horizontal sum is performed in a serial manner.

// compile with: gcc -mavx -Wall -O2 sum-float-avx.c -o sum-float-avx -DN=10000 -DR=10000

// NOTE1: float result may not have the accuracy required to store sums with large N
// NOTE2: order of additions is not the same as in serial reduction
// NOTE3: N must be a multiple of 8



#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <immintrin.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
  double ts,te;
  float *a,partial_sums[8],result;
  __m256 *pa,sum;
  
  // 1. allocate array (aligned to 32 bytes)
  int i = posix_memalign((void **)&a,32,N*sizeof(float));
  if (i!=0) {
    printf("Allocation failed!\n");
    exit(1);
  }	  
  
  // 2. init array to random int values, from 0 to 9 (represented exactly as floats)
  for (int i=0;i<N;i++) {
    a[i] = rand()%10;
  }


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // alias sse ptr to float array start
  pa = (__m256 *)a;
    
  // 3. reduce array to sum (R times)
  for (int j=0;j<R;j++) {
    sum = _mm256_setzero_ps();
    for (int i=0;i<N/8;i++) {
      sum = _mm256_add_ps(sum,pa[i]);
    }

    // move 8-float sum to result = unaligned store, slow
    _mm256_storeu_ps(partial_sums,sum);
    
    // serially add 8 parts to final result
    result = 0.0;
    for (int i=0;i<8;i++) {
      result += partial_sums[i];
    }

  }
    
  // get ending time
  get_walltime(&te);
  
  // 4. check result
  float check = 0.0;
  for (int i=0;i<N;i++) {
    check += a[i];
  }
  if (check!=result) {	// NOTE: this comparison may fail for large values of N
    printf("Error! found %f instead of %f\n",result,check);
  }
  
  // 5. free array
  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
