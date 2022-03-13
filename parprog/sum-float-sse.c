// sum reduction of an array of floats using sse instructions

// compile with: gcc -msse2 -Wall -O2 sum-float-sse.c -o sum-float-sse -DN=1000 -DR=100000

// NOTE1: float result has not the accuracy required by big values of N!
// NOTE2: order of additions is NOT the same as in serial reduction (may lead to different result)!
// NOTE3: N must be a multiple of 4!

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <emmintrin.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}




int main() {
  double ts,te;
  float *a,fsum;
  
  __m128 *pa,sum,t1,t2;
  
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
  
  // alias sse ptr to float array start
  pa = (__m128 *)a;
  
  // 3. reduce array to sum (R times)
  for (int j=0;j<R;j++) {
    sum = _mm_set1_ps(0);
    for (int i=0;i<N/4;i++) {
      sum = _mm_add_ps(sum,pa[i]);
    }
    // perform horizontal sum
    t1 = _mm_shuffle_ps(sum,sum,_MM_SHUFFLE(2,3,0,1));	// C D A B
    t2 = _mm_add_ps(sum,t1);				// D+C C+D B+A A+B
    t1 = _mm_movehl_ps(t1,t2);				// C D D+C C+D
    t2 = _mm_add_ss(t1,t2);				// C D D+C A+B+C+D
  }
  
  // transfer result to float
  fsum = _mm_cvtss_f32(t2);	// float _mm_cvtss_f32 (__m128 a)
  
  // get ending time
  get_walltime(&te);
  
  // 4. print result (no check due to float rounding)
  printf("Result = %f\n",fsum);

  // 5. free array
  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
