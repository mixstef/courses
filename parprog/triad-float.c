#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// compile with: gcc -Wall -O2 triad-float.c -o triad-float -DN=10000 -DR=10000

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

int main() {
float *a,*b,*c,*d;
double ts,te,mflops;

  // allocate test arrays
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) { printf("Allocation failed!\n"); exit(1); }
  b = (float *)malloc(N*sizeof(float));
  if (b==NULL) { printf("Allocation failed!\n"); free(a); exit(1); }
  c = (float *)malloc(N*sizeof(float));
  if (c==NULL) { printf("Allocation failed!\n"); free(a); free(b); exit(1); }
  d = (float *)malloc(N*sizeof(float));
  if (d==NULL) { printf("Allocation failed!\n"); free(a); free(b); free(c); exit(1); }
  
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
    for (int i=0;i<N;i++) {
      a[i] = b[i]*c[i]+d[i];
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
 
  // compute mflops/sec (2 floating point operation per R*N passes)
  mflops = (R*N*2.0)/((te-ts)*1e6);
  
  printf("MFLOPS/sec = %f\n",mflops);
  
  // free arrays
  free(a); free(b); free(c); free(d);
  
  return 0;
}

