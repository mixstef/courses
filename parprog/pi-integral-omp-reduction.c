// Example calculating pi value using integration. OpenMP reduction version.
// Compile with: gcc -O2 -Wall -fopenmp pi-integral-omp-reduction.c -o pi-integral-omp-reduction -DN=10000000


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
double pi,w,sum,x;
double ts,te;


  w = 1.0/N;	// integration step


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  sum = 0.0;
  #pragma omp parallel for private(x) reduction(+:sum)
  for (int i=1;i<=N;i++) {
    x = w*(i-0.5);	// midpoint
    sum += 4.0/(1.0+x*x); // NOTE: without mult by step (w), done later
  }
 
  pi = w*sum;

  // get ending time
  get_walltime(&te);

  
  printf("Computed pi=%.10f\n",pi);
  printf("Exec Time (sec) = %f\n",te-ts);

  return 0;
}
