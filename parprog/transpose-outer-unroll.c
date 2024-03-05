// Code example to transpose a NxN matrix, outer loop unrolling
// compile with:  gcc -Wall -O2 transpose-outer-unroll.c -o transpose-outer-unroll -DN=4000
// NOTE: N must be multiple of 4


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
unsigned int i,j;
double *a,*b;
double ts,te,aps;


  a = (double *)malloc(N*N*sizeof(double)); 
  if (a==NULL) {
    printf("alloc error!\n");
    exit(1);
  }

  b = (double *)malloc(N*N*sizeof(double)); 
  if (b==NULL) {
    printf("alloc error!\n");
    free(a);
    exit(1);
  }

  // warmup
  for (i=0;i<N*N;i++) {
     a[i] = 2.0*i;
     b[i] = -i;
  } 

  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // transpose workload, unroll outer loop
  for (i=0;i<N;i+=4) {
    for (j=0;j<N;j++) {
      a[j*N+i] = b[i*N+j];
      a[j*N+i+1] = b[(i+1)*N+j];
      a[j*N+i+2] = b[(i+2)*N+j];
      a[j*N+i+3] = b[(i+3)*N+j];
    }
  }

  // get ending time
  get_walltime(&te);

  // check operation
  int done = 0;
  for (i=0;i<N && done!=1;i++) {
    for (j=0;j<N;j++) {
      if (a[j*N+i] != b[i*N+j]) {
        printf("Error!\n");
        done = 1;
        break;
      }
    }
  }

  
  // compute avg array element accesses /sec (total NROWSxNCOLSx(1load+1store) element accesses)
  aps = (2.0*N*N)/((te-ts)*1e6);
  
  printf("avg array element Maccesses/sec = %f\n",aps);

  free(b);
  free(a);

  return 0;
}

