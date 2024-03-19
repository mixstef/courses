// compile with:
// gcc -mavx -Wall -O2 triad-float-avx.c -o triad-float-avx -DN=10000 -DR=10000
// NOTE: N must be multiple of 8!


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <immintrin.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

int main() {
float *a,*b,*c,*d;
__m256 *pa,*pb,*pc,*pd;
double ts,te,mflops;
int i;

  // allocate test arrays
  // allocate test arrays - request alignment at 32 bytes
  i = posix_memalign((void **)&a,32,N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); exit(1); }
  i = posix_memalign((void **)&b,32,N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); free(a); exit(1); }
  i = posix_memalign((void **)&c,32,N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); free(a); free(b); exit(1); }
  i = posix_memalign((void **)&d,32,N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); free(a); free(b); free(c); exit(1); }

  // alias the sse pointers to arrays
  pa = (__m256 *)a;    
  pb = (__m256 *)b;
  pc = (__m256 *)c;
  pd = (__m256 *)d;
    
  //initialize all arrays - cache warm-up
  for (int i=0;i<N;i++) {
    a[i] = 2.0*i;
    b[i] = -i;
    c[i] = i+5.0;
    d[i] = i;
  }
 
  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // do artificial work
  for (int j=0;j<R;j++) {  
    for (int i=0;i<N/8;i++) {
      pa[i] = _mm256_add_ps(_mm256_mul_ps(pb[i],pc[i]),pd[i]);
    }
  }
 
  // get ending time
  get_walltime(&te);
  
  // check result - avoid loop removal by compiler
   for (int i=0;i<N;i++) {
    if (a[i]!=b[i]*c[i]+d[i]) {
      printf("Error!\n");
      break;
    }
  }
 
  // compute mflops/sec (2 floating point operations per R*N passes)
  mflops = (R*N*2.0)/((te-ts)*1e6);
  
  printf("MFLOPS/sec = %f\n",mflops);
  
  // free arrays
  free(a); free(b); free(c); free(d);
  
  return 0;
}

