// Template for AVX PS (8 single-precision floats) tests.

// Compile with: gcc -mavx -O2 avx-template.c -o avx-template


#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h>



int main() {

float *a,*b,*c;
__m256 *pa,*pb,*pc;

int i;

  // allocate test arrays - request alignment at 32 bytes
  i = posix_memalign((void **)&a,32,8*sizeof(float));
  if (i!=0) exit(1);
  i = posix_memalign((void **)&b,32,8*sizeof(float));
  if (i!=0) { free(a); exit(1); }
  i = posix_memalign((void **)&c,32,8*sizeof(float));
  if (i!=0) { free(b); free(a); exit(1); }
  
  //initialize float input test arrays a, b
  for (i=0;i<8;i++) {
    a[i] = i;
    b[i] = i+(float)i/10;
  }
  
  // alias the avx pointers to input and output test arrays
  pa = (__m256 *)a; pb = (__m256 *)b;
  pc = (__m256 *)c;
  
  // test avx instructions here
  // e.g. ADD
    *pc = _mm256_add_ps(*pa,*pb);
  
  // print result of output array fc
  for (i=0;i<8;i++) {
    printf("a[%d]= %f   b[%d]= %f   c[%d]= %f\n",i,a[i],i,b[i],i,c[i]);
  }

   // free arrays
  free(c); free(b); free(a); 
  return 0;
}
