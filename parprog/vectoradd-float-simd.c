// OpenMP SIMD map example
// compile with: gcc -msse2 -O2 -Wall -fopenmp vectoradd-float-simd.c -o vectoradd-float-simd -DN=10000 -DR=10000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <omp.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

int main() {
float *a,*b,*c;
double ts,te,mflops;
int i;

  // allocate test arrays
  // allocate test arrays - request alignment at 16 bytes
  i = posix_memalign((void **)&a,16,N*sizeof(float));
  if (i!=0) exit(1);
  i = posix_memalign((void **)&b,16,N*sizeof(float));
  if (i!=0) { free(a); exit(1); }
  i = posix_memalign((void **)&c,16,N*sizeof(float));
  if (i!=0) { free(a); free(b); exit(1); }

  
  //initialize all arrays - cache warm-up
  for (int i=0;i<N;i++) {
    a[i]=2.0*i;
    b[i]=-i;
    c[i]=i+5.0;
  }
 
  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // do artificial work
  for (int j=0;j<R;j++) {
  
    #pragma omp simd aligned(a,b,c:16)  
    for (int i=0;i<N;i++) {
      a[i] = b[i]+c[i];
    }
  }
 
  // get ending time
  get_walltime(&te);
  
  // check result - avoid loop removal by compiler
   for (int i=0;i<N;i++) {
    if (a[i]!=b[i]+c[i]) {
      printf("Error!\n");
      break;
    }
  }
 
  // compute mflops/sec (1 floating point operation per R*N passes)
  mflops = (R*N)/((te-ts)*1e6);
  
  printf("MFLOPS/sec = %f\n",mflops);
  
  // free arrays
  free(a); free(b); free(c);
  
  return 0;
}

