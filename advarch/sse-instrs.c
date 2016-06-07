#include <stdio.h>
#include <stdlib.h>

#include <emmintrin.h>

#define FN 4
float testvala[] = { 1.0,2.0,3.0,4.0 };
float testvalb[] = { 1.1,2.2,3.3,4.4 };


// Tests various SSE/SSE2 instructions
// no timing tests, only operation examples

// Compile with: gcc -msse2 -O sse-instrs.c -o sse-instrs
// NOTE: on 32-bit arch, -msse2 MUST be specified!

// To check assembly, compile with: gcc -msse2 -O sse-instrs.c -S -masm=intel

int main() {

// for SSE PS (single-precision FP) tests 
float *fa,*fb,*fc;
__m128 *vfa,*vfb,*vfc;

int i;

  // allocate test arrays - request alignment at 16 bytes
  i = posix_memalign((void **)&fa,16,FN*sizeof(float));
  if (i!=0) exit(1);
  i = posix_memalign((void **)&fb,16,FN*sizeof(float));
  if (i!=0) { free(fa); exit(1); }
  i = posix_memalign((void **)&fc,16,FN*sizeof(float));
  if (i!=0) { free(fb); free(fa); exit(1); }
  
  //initialize float test arrays
  for (i=0;i<FN;i++) {
    *(fa+i) = testvala[i];
    *(fb+i) = testvalb[i];
  } 

  // alias the sse pointers to arrays
  vfa = (__m128 *)fa; vfb = (__m128 *)fb;
  vfc = (__m128 *)fc;


  // ADD (same is SUB) also MIN, MAX (select each of 4) and logical operations
  *vfc = _mm_add_ps(*vfa,*vfb);
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]);	// 2.100000 4.200000 6.300000 8.400000

  // MUL (same is DIV)
  *vfc = _mm_mul_ps(*vfa,*vfb);
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]);	// 1.100000 4.400000 9.900000 17.600000

  // SQRT (same is RCP (reciprocal) and RSQRT (reciprocal sqrt))
  *vfc = _mm_sqrt_ps(*vfa);
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]);	// 1.000000 1.414214 1.732051 2.000000


  // various compare operations, leaves mask of 1s or 0s in relevant positions
  *vfc = _mm_cmpeq_ps(*vfa,*vfb);
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]);	// 0.000000 0.000000 0.000000 0.000000

  // shuffling instr 
  // unshuffled order is 3,2,1,0 (left to right) - first 2 args select from vfb,last 2 from vfa
  // i.e. with
  // __m128 out = _mm_shuffle_ps(__m128 lo,__m128 hi,_MM_SHUFFLE(hi3,hi2,lo1,lo0))
  // result is
  //  out[0]=lo[lo0];
  //  out[1]=lo[lo1];
  //  out[2]=hi[hi2];
  //  out[3]=hi[hi3];
  *vfc = _mm_shuffle_ps(*vfa,*vfb,_MM_SHUFFLE(3,1,0,1)); 
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]); // this will print 2.000000 1.000000 2.200000 4.400000

  // set to a quad floats - also: set1 sets all to one float (see below)
  *vfc = _mm_set_ps(22.2,10.11,33.3,7.123);
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]);

  // unpack lower 2 (left in init values) of A and B and interleave in dest - also : unpack high
  *vfc = _mm_unpacklo_ps(*vfa,*vfb);
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]);	// 1.000000 1.100000 2.000000 2.200000

  // movlh = move lower 2 of A to lower (left) of dest, lower 2 of B to upper (right) of dest
  //- also movhl
  *vfc = _mm_set1_ps(0.0);
  *vfc = _mm_movelh_ps(*vfa,*vfb);
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]);	// 1.000000 2.000000 1.100000 2.200000
  

  // explicit store, also: storeu (un-aligned) and stream (non temporal store)
  // similar with loads
  _mm_store_ps(fc,*vfb);
  printf("%f %f %f %f\n",fc[0],fc[1],fc[2],fc[3]);

   // free arrays
  free(fc); free(fb); free(fa); 
  return 0;
}
