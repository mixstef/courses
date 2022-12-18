#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// compile with: gcc -Wall -O2 vectoradd.c -o vectoradd -DN=1000000 -DR=10000
// test also with R=1

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

int main() {
double *a,*b,*c;
double ts,te,mflops;

  // allocate test arrays
  a = (double *)malloc(N*sizeof(double));
  if (a==NULL) exit(1);
  b = (double *)malloc(N*sizeof(double));
  if (b==NULL) { free(a); exit(1); }
  c = (double *)malloc(N*sizeof(double));
  if (c==NULL) { free(a); free(b); exit(1); }
  
  //initialize all arrays - cache warm-up
  for (int i=0;i<N;i++) {
    a[i]=0.0;
    b[i]=-i;
    c[i]=i+5.0;
  }
 
  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // do artificial work
    for (int i=0;i<N;i++) {
      for (int j=0;j<R;j++) {
        a[i] += b[i]+c[i];
      }
    }
 
  // get ending time
  get_walltime(&te);
  
  // check result - avoid loop removal by compiler
   for (int i=0;i<N;i++) {
    if (a[i]!=(b[i]+c[i])*R) {
      printf("Error!\n");
      break;
    }
  }
 
  // compute mflops/sec (2 double floating point operations per R*N passes)
  mflops = (2.0*R*N)/((te-ts)*1e6);
  
  printf("MFLOPS/sec = %f\n",mflops);
  
  // free arrays
  free(a); free(b); free(c);
  
  return 0;
}

