#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <emmintrin.h>

// Compile with: gcc -msse2 -O2 -Wall float-triad-sse.c -o float-triad-sse -DN=.. -DR=..


// use -DN=.. and -DR=... at compilation
//#define N 1000000
//#define R 10

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

int main() {
float *a,*b,*c,*d;
__m128 *va,*vb,*vc,*vd;

int i,j;
double ts,te,mflops;

  // allocate test arrays - request alignment at 16 bytes
  i = posix_memalign((void **)&a,16,N*sizeof(float));
  if (i!=0) exit(1);
  i = posix_memalign((void **)&b,16,N*sizeof(float));
  if (i!=0) { free(a); exit(1); }
  i = posix_memalign((void **)&c,16,N*sizeof(float));
  if (i!=0) { free(a); free(b); exit(1); }
  i = posix_memalign((void **)&d,16,N*sizeof(float));
  if (i!=0) { free(a); free(b); free(c); exit(1); }
  
  //initialize all arrays - cache warm-up
  for (i=0;i<N;i++) {
    a[i]=0.0; b[i]=1.0; c[i]=2.0; d[i]=3.0;
  }
 
 
  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // do triad artificial work
  for (j=0;j<R;j++) {
    // alias the sse pointers to arrays
    va = (__m128 *)a; vb = (__m128 *)b;
    vc = (__m128 *)c; vd = (__m128 *)d;

    for (i=0;i<N;i+=4) {
      *va = _mm_add_ps(*vb,_mm_mul_ps(*vc,*vd));
      va++; vb++; vc++; vd++;
      //a[i] = b[i]+c[i]*d[i];
    }
  }
 
  // get ending time
  get_walltime(&te);
  
  // compute mflops/sec (2 operations per R*N passes)
  mflops = (R*N*2.0)/((te-ts)*1e6);
  
  printf("MFLOPS/sec = %f\n",mflops);
  
  // free arrays
  free(a); free(b); free(c); free(d);
  
  return 0;
}
