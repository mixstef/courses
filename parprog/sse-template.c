// Template for SSE PS (4 single-precision floats) tests.

// Compile with: gcc -msse2 -O2 sse-template.c -o sse-template
// NOTE: -msse2 is optional (on 32-bit arch, -msse2 MUST be specified)


#include <stdio.h>
#include <stdlib.h>

#include <emmintrin.h>



int main() {

float *fa,*fb,*fc;

int i;

  // allocate test arrays - request alignment at 16 bytes
  i = posix_memalign((void **)&fa,16,4*sizeof(float));
  if (i!=0) exit(1);
  i = posix_memalign((void **)&fb,16,4*sizeof(float));
  if (i!=0) { free(fa); exit(1); }
  i = posix_memalign((void **)&fc,16,4*sizeof(float));
  if (i!=0) { free(fb); free(fa); exit(1); }
  
  //initialize float input test arrays fa, fb
  fa[0] = 1.0; fa[1] = 2.0; fa[2] = 3.0; fa[3] = 4.0;
  fb[0] = 1.1; fb[1] = 2.2; fb[2] = 3.3; fb[3] = 4.4;
  
  // test sse instructions here
  // TODO
  
  // print result of output array fc
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]);

   // free arrays
  free(fc); free(fb); free(fa); 
  return 0;
}
