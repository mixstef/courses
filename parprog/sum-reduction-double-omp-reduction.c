// OpenMP sum-reduction example, using reduction clause.
// Compile with: gcc -O2 -Wall -fopenmp sum-reduction-double-omp-reduction.c -o sum-reduction-double-omp-reduction -DN=10000000

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
double ts,te;
double *a;

  // allocate array
  a = (double *)malloc(N*sizeof(double));
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
  
  // reduce array to sum
  double sum = 0;
  
  #pragma omp parallel for reduction(+:sum)
  for (int i=0;i<N;i++) {
    sum += a[i];
  }
    
  // get ending time
  get_walltime(&te);

  // check result
  double result = ((double)N*(N+1))/2;  
  if (sum!=result) {
    printf("Reduction error!\n");
  }

  // free array
  free(a);
 
  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
