// Example implementation of horizontal sum for AVX packed floats 

// compile with: gcc -mavx -Wall -O2 float-hsum-avx.c -o float-hsum-avx

#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h>


int main() {

float *a,fsum;
__m256 *pa;


int i;

  // allocate test array - request alignment at 32 bytes
  i = posix_memalign((void **)&a,32,8*sizeof(float));
  if (i!=0) exit(1);
  
  //initialize float input test array
  for (i=0;i<8;i++) {
    a[i] = rand() % 100;
    printf("a[%d] = %f\n",i,a[i]);
  }

  pa = (__m256 *)a;
  
  // horizontal sum of the 8 packed floats, *pa = h g f e d c b a
  
  // add high and low 128-bit halves into a __m128 number t1 = h+d g+c f+b e+a 
  __m128 t1 = _mm_add_ps(_mm256_extractf128_ps(*pa,1),_mm256_castps256_ps128(*pa));
  // create t2, a shuffled version of t1,  t2 = g+c h+d e+a f+b
  __m128 t2 = _mm_shuffle_ps(t1,t1,_MM_SHUFFLE(2,3,0,1)); 
  // add t1 and t2, result t3 = h+d+g+c g+c+h+d f+b+e+a e+a+f+b    
  __m128 t3 = _mm_add_ps(t1,t2);
  // set t2 to upper parts of t2 and t3, now t2 = g+h h+d h+d+g+c g+c+h+d 
  t2 = _mm_movehl_ps(t2,t3);
  // add lowest parts of t2 and t3, now t3 = g+h h+d h+d+g+c g+c+h+d+e+a+f+b  
  t3 = _mm_add_ss(t2,t3);
  // copy to a single float
  fsum = _mm_cvtss_f32(t3); 

  // check result
  printf("Sum = %f\n",fsum);
  float tsum = 0.0;
  for (i=0;i<8;i++) {
    tsum += a[i];
  }

  if (fsum==tsum) {
    printf("ok\n");
  }

   // free array
  free(a);
  return 0;
}
