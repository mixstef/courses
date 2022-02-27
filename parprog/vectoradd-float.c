#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// compile with: gcc -Wall -O2 vectoradd-float.c -o vectoradd-float -DN=10000000 -DR=10

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

int main() {
float *a,*b,*c;
double ts,te,mflops;

  // allocate test arrays
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) exit(1);
  b = (float *)malloc(N*sizeof(float));
  if (b==NULL) { free(a); exit(1); }
  c = (float *)malloc(N*sizeof(float));
  if (c==NULL) { free(a); free(b); exit(1); }
  
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

