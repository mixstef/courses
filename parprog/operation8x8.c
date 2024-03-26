// Sample 8x8 float matrix operation, using SSE/AVX instructions

// Compile with: gcc -mavx -O2 operation8x8.c -o operation8x8



#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h>



int main() {

float *a,*b;
__m128 *pa;	// sse (128-bit) pointer
__m256 *pb;	// avx (256-bit) pointer

int i;

  // allocate test arrays - request alignment at 32 bytes
  i = posix_memalign((void **)&a,32,8*8*sizeof(float));
  if (i!=0) exit(1);
  i = posix_memalign((void **)&b,32,8*8*sizeof(float));
  
  //initialize float input test array a
  for (i=0;i<8*8;i++) {
    a[i] = i;
  }
  
  // alias the sse/avx pointers to input and output test arrays
  pa = (__m128 *)a; pb = (__m256 *)b;
  
  // operation, from a to b
  for (int i=0;i<2;i++) {
    __m256 t1 = _mm256_insertf128_ps(_mm256_castps128_ps256(pa[0+i]),pa[8+i],1);
    __m256 t2 = _mm256_insertf128_ps(_mm256_castps128_ps256(pa[2+i]),pa[10+i],1);
    __m256 u1 = _mm256_unpacklo_ps(t1,t2);
    __m256 u2 = _mm256_unpackhi_ps(t1,t2);
  
    __m256 t3 = _mm256_insertf128_ps(_mm256_castps128_ps256(pa[4+i]),pa[12+i],1);
    __m256 t4 = _mm256_insertf128_ps(_mm256_castps128_ps256(pa[6+i]),pa[14+i],1);
    __m256 u3 = _mm256_unpacklo_ps(t3,t4);
    __m256 u4 = _mm256_unpackhi_ps(t3,t4);
  
    pb[4*i+0] = _mm256_shuffle_ps(u1,u3,0x44);
    pb[4*i+1] = _mm256_shuffle_ps(u1,u3,0xee);
    pb[4*i+2] = _mm256_shuffle_ps(u2,u4,0x44);
    pb[4*i+3] = _mm256_shuffle_ps(u2,u4,0xee);
  }
  

  // free arrays
  free(b); free(a); 
  return 0;
}
