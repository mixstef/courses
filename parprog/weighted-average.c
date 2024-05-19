// Weighted average sample.
// Compile with: gcc -O2 -Wall weighted-average.c -o weighted-average -DN=10000000

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
double *x,*w;

  // allocate arrays
  x = (double *)malloc(N*sizeof(double));
  if (x==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  
  w = (double *)malloc(N*sizeof(double));
  if (w==NULL) {
    printf("Allocation failed!\n");
    free(x);
    exit(1);
  }	  

  // random init arrays
  for (int i=0;i<N;i++) {
    x[i] = (double)rand()/RAND_MAX;
    w[i] = (double)rand()/RAND_MAX;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // compute weighted average
  double sum = 0;
  double wsum = 0;
  for (int i=0;i<N;i++) {
    sum += x[i]*w[i];
    wsum += w[i];
  }
  
  double wavg = sum/wsum;

  // get ending time
  get_walltime(&te);


  // free arrays
  free(w);
  free(x);

  // print weighted average and exec time
  printf("Weighted average = %f\n",wavg);
  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
