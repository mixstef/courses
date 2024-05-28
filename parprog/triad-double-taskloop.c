// Triad benchmark with arrays of doubles, taskloop OpenMP implementation
// compile with: gcc -Wall -O2 -fopenmp triad-double-taskloop.c -o triad-double-taskloop -DN=10000 -DR=10000


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
double *a,*b,*c,*d;
double ts,te,mflops;

  // allocate test arrays
  a = (double *)malloc(N*sizeof(double));
  if (a==NULL) exit(1);
  b = (double *)malloc(N*sizeof(double));
  if (b==NULL) { free(a); exit(1); }
  c = (double *)malloc(N*sizeof(double));
  if (c==NULL) { free(a); free(b); exit(1); }
  d = (double *)malloc(N*sizeof(double));
  if (d==NULL) { free(a); free(b); free(c); exit(1); }
  
  //initialize all arrays - cache warm-up
  for (int i=0;i<N;i++) {
    a[i]=2.0*i;
    b[i]=-i;
    c[i]=i+5.0;
    d[i]=-7.0*i;
  }
 
  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // do triad artificial work
  #pragma omp parallel
  {
  
    #pragma omp single nowait
    {

      for (int j=0;j<R;j++) {
  

        #pragma omp taskloop
        for (int i=0;i<N;i++) {
          a[i] = b[i]*c[i]+d[i];
        }
              
      }
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
  mflops = (2.0*R*N)/((te-ts)*1e6);
  
  printf("MFLOPS/sec = %f\n",mflops);
  
  // free arrays
  free(a); free(b); free(c); free(d);
  
  return 0;
}

