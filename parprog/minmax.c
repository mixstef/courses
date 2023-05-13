// Simple min/max reduction example.
// Compile with: gcc -O2 -Wall minmax.c -o minmax

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#define N 100000000


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

int main() {
double ts,te;
double *a;

  // allocate array
  a = (double *)malloc(N*sizeof(double));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  

  // fill array with random numbers
  srand(0);
  a[0] = (double)rand()/RAND_MAX;
  double checkmin = a[0];
  double checkmax = a[0];
  for (size_t i=1;i<N;i++) {
    a[i] = (double)rand()/RAND_MAX;
    if (a[i]<checkmin) checkmin = a[i];
    if (a[i]>checkmax) checkmax = a[i];    
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // min/max reduction
  double minval = a[0];
  double maxval = a[0];
  for (size_t i=1;i<N;i++) {
    if (a[i]<minval) minval = a[i];
    if (a[i]>maxval) maxval = a[i];
  }

  // get ending time
  get_walltime(&te);

  // check result
  if ((minval!=checkmin)||(maxval!=checkmax)) {
    printf("Reduction error!\n");
  }

  // free array
  free(a);
 
  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
