// Scans all elements of a float array and clamps values to a threshold, producing a new array as output - SSE version

// compile with: gcc -msse2 -Wall -O2 threshold-clamp-float-sse.c -o threshold-clamp-float-sse -DN=10000 -DR=10000 -DTHRESHOLD=50.0

// NOTE: N must be a multiple of 4!
// NOTE2: performance is insensitive to THRESHOLD value


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
  float *a,*b;
  __m128 *pa,*pb;
  
  // 1. allocate arrays
  int i = posix_memalign((void **)&a,16,N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); exit(1); }
  i = posix_memalign((void **)&b,16,N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); free(a); exit(1); }

  // alias m128 ptrs to arrays
  pa = (__m128 *)a;
  pb = (__m128 *)b;
  
  // 2. init a array to random int values, from 0 to 99 (represented exactly as floats)
  for (int i=0;i<N;i++) {
    a[i] = rand()%100;
    b[i] = -i;	// init b array to negative numbers
  }


  // get starting time (double, seconds) 
  get_walltime(&ts);

  // 4-float packed thresholds
  __m128 thres4 = _mm_set1_ps(THRESHOLD);
  
  // 3. scan array (R times)
  for (int j=0;j<R;j++) {

    for (int i=0;i<N/4;i++) {
      
      __m128 mask = _mm_cmpgt_ps(pa[i],thres4);
      
      pb[i] = _mm_or_ps(_mm_and_ps(mask,thres4),_mm_andnot_ps(mask,pa[i]));

    }
  }
  
  // get ending time
  get_walltime(&te);
  
  // 4. check result
  for (int i=0;i<N;i++) {
    if ((a[i]<=THRESHOLD && b[i]!=a[i])||(a[i]>THRESHOLD && b[i]!=THRESHOLD)) {
      printf("Error!\n");
      break;
    }
  }
  
  // 5. free arrays
  free(a);
  free(b);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
