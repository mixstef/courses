#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

// compile with: gcc -fopenmp -O2 -Wall pi-integral-omp.c -o pi-integral-omp


#define N 1000000	// integration steps

int main() {
double pi,w,sum,x;

  w = 1.0/N;	// integration step
  pi = 0.0;
  sum = 0.0;
  
  #pragma omp parallel firstprivate(sum)
  {
  
    #pragma omp for private(x) nowait
    for (int i=1;i<=N;i++) {
      x = w*(i-0.5);	// midpoint
      sum += 4.0/(1.0+x*x); // NOTE: without mult by step (w), done later
    }
 
    #pragma omp critical
    {
      pi += w*sum;
    }
    
  }
  
  printf("Computed pi=%.10f\n",pi);
  return 0;
}
