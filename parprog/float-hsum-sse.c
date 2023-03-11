// Example implementation of horizontal sum for SSE2 packed floats 

// compile with: gcc -msse2 -Wall -O2 float-hsum-sse.c -o float-hsum-sse

#include <stdio.h>
#include <stdlib.h>

#include <emmintrin.h>


int main() {

float *a,fsum;
__m128 *pa;


int i;

  // allocate test array - request alignment at 16 bytes
  i = posix_memalign((void **)&a,16,4*sizeof(float));
  if (i!=0) exit(1);
  
  //initialize float input test array
  for (i=0;i<4;i++) {
    a[i] = rand() % 100;
    printf("%f\n",a[i]);
  }

  pa = (__m128 *)a;
  
  // horizontal sum of the 4 packed floats
  __m128 t1 = _mm_shuffle_ps(*pa,*pa,_MM_SHUFFLE(2,3,0,1)); // C D A B
  __m128 t2 = _mm_add_ps(*pa,t1);			    // D+C C+D B+A A+B
  t1 = _mm_movehl_ps(t1,t2);				    // C D D+C C+D
  t2 = _mm_add_ss(t1,t2);				    // C D D+C A+B+C+D
  fsum = _mm_cvtss_f32(t2); 

  // check result
  printf("Sum = %f\n",fsum);
  float tsum = 0.0;
  for (i=0;i<4;i++) {
    tsum += a[i];
  }

  if (fsum==tsum) {
    printf("ok\n");
  }

   // free array
  free(a);
  return 0;
}
