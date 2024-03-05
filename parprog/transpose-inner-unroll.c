// Code example to transpose a NxN matrix, inner loop unrolling
// compile with:  gcc -Wall -O2 transpose-inner-unroll.c -o transpose-inner-unroll -DN=4000
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
  
  // transpose workload, unroll inner loop
  for (i=0;i<N;i++) {
    for (j=0;j<N;j+=4) {
      a[j*N+i] = b[i*N+j];
      a[(j+1)*N+i] = b[i*N+j+1];
      a[(j+2)*N+i] = b[i*N+j+2];
      a[(j+3)*N+i] = b[i*N+j+3];
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

