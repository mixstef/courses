// In-place serial LU decomposition of a square matrix
// Compile with:  gcc -Wall -O2 lu-serial.c -o lu-serial -DN=1056


#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>



void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


void serial_lu(double *a) {

  for (int i=0;i<N;i++) {
  
    for (int j=0;j<i;j++) {
      for (int k=0;k<j;k++) {
        a[i*N+j] -= a[i*N+k] * a[k*N+j];
      }
      a[i*N+j] /= a[j*N+j];
    }
    
    for (int j=i;j<N;j++) {
      for (int k=0;k<i;k++) {
        a[i*N+j] -= a[i*N+k]*a[k*N+j];
      }
    }
  }

}


void your_lu(double *a) {

  // replace next with your code
  serial_lu(a);

}


int main() {
double ts,te;

double *a,*acheck;

  // allocate input/output and check arrays
  a = (double *)malloc(N*N*sizeof(double));
  if (a==NULL) {
    exit(1);
  }
  acheck = (double *)malloc(N*N*sizeof(double));
  if (acheck==NULL) {
    free(a);
    exit(1);
  }
  
  // init elements of arrays
  for (int row=0;row<N;row++) {
    double sum = 0.0;
    for (int col=0;col<N;col++) {
      if (row!=col) {
        sum += a[row*N+col] = acheck[row*N+col] = (double)rand()/RAND_MAX;
      }
    }
    a[row*N+row] = acheck[row*N+row] = sum + 0.1;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  // workload (LU decomposition)
  your_lu(a);

  // get ending time
  get_walltime(&te);

  // print computation time
  printf("Computation time = %f sec\n",(te-ts));

  // check result
  serial_lu(acheck);
  int success = 1;
  for (int i=0;i<N*N;i++) {
    if (a[i]!=acheck[i]) {
      success = 0;
      break;
    }
  }
  if (success==1) printf("Success!\n");
  else printf("Error!\n");
  
  // free arrays
  free(acheck); 
  free(a);
  
  return 0;
}
