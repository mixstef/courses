// Simple prefix sum example.
// Compile with: gcc -O2 -Wall prefix-sum-double.c -o prefix-sum-double -DN=10000000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>



void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
  
  double ts,te;

  // allocate array
  double *a = (double *)malloc(N*sizeof(double));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  

  // init array to 1..N
  for (int i=0;i<N;i++) {
    a[i] = i+1;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // compute (inclusive) prefix sum, in-place
  double sum = 0;
  for (int i=0;i<N;i++) {
    sum += a[i];
    a[i] = sum;
  }

  // get ending time
  get_walltime(&te);

  // check result
  for (int i=0;i<N;i++) {
    if (a[i]!=((double)(i+1)*(i+2))/2) {
      printf("Prefix sum error!\n");
      break;
    }
  }

  // free array
  free(a);
 
  printf("Exec Time (sec) = %f\n",te-ts);

  
  return 0;
}
