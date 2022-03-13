#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <emmintrin.h>

// compile with: gcc -msse2 -Wall -O2 filter-int-sse.c -o filter-int-sse -DN=10000000 -DR=10
// NOTE: N must be multiple of 4!


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}



int main() {
  int *a,*b;
  __m128i *pa,*pb;
  double ts,te;
    
  int t = posix_memalign((void **)&a,16,N*sizeof(int));  
  if (t!=0) { printf("alloc error\n"); exit(1); }
  t = posix_memalign((void **)&b,16,N*sizeof(int));  
  if (t!=0) { free(a); printf("alloc error\n"); exit(1); }
  
  for (int i=0;i<N;i++) {
    a[i] = i % 100;
    b[i] = -2*i;
  }

  // alias the sse pointers to arrays
  pa = (__m128i *)a;
  pb = (__m128i *)b;
  
  // the threshold constant
  __m128i threshold = _mm_set1_epi32(50);
  
  // the output value if not zero (=1)
  __m128i output = _mm_set1_epi32(1);

  // get starting time (double, seconds) 
  get_walltime(&ts);

  for (int j=0;j<R;j++) {  
    for (int i=0;i<N/4;i++) {
      pb[i] = _mm_and_si128(_mm_cmpgt_epi32(pa[i],threshold),output);
    }
  } 

  // get ending time
  get_walltime(&te);

  for (int i=0;i<N;i++) {
    if (((a[i]>50)&&(b[i]!=1))||((a[i]<=50)&&(b[i]!=0))) {
      printf("error\n");
      break;
    }
  }
  
  free(b);
  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);

  return 0;
}


