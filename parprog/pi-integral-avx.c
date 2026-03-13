// Calculates pi value using integration with AVX instructions.
// Compile with: gcc -mavx -O2 -Wall pi-integral-avx.c -o pi-integral-avx -DN=10000000

// NOTE: N must be a multiple of 4

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
double result,partial_sums[4];
double pi,w;
double ts,te;


  w = 1.0/N;	// integration step


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // constant setup
  __m256d w4 = _mm256_set1_pd(w);
  __m256d c4 = _mm256_set1_pd(4.0);
  __m256d c1 = _mm256_set1_pd(1.0);
  
  // init accumulator to 0  
  __m256d sum = _mm256_setzero_pd();
  // init counter (i) to 1..4 (with 0.5 pre-subtracted!)
  __m256d cnt = _mm256_set_pd(3.5,2.5,1.5,0.5);
  
  for (int i=1;i<=N/4;i++) {
    
    // x = w*(i-0.5);	// midpoint
    __m256d x = _mm256_mul_pd(w4,cnt);
    
    // sum += 4.0/(1.0+x*x); // NOTE: without mult by step (w), done later
    sum = _mm256_add_pd(sum,_mm256_div_pd(c4,_mm256_add_pd(c1,_mm256_mul_pd(x,x))));
    
    // step counter to +4 values
    cnt = _mm256_add_pd(cnt,c4);
  }

  // move 4-double sum to result = unaligned store, slow
  _mm256_storeu_pd(partial_sums,sum);
  
  // serially add 4 parts to final result
  result = 0.0;
  for (int i=0;i<4;i++) {
    result += partial_sums[i];
  }  
 
  pi = w*result;

  // get ending time
  get_walltime(&te);

  
  printf("Computed pi=%.10f\n",pi);
  printf("Exec Time (sec) = %f\n",te-ts);

  return 0;
}
