// Example of firstprivate clause usage

// Compile with: gcc -O2 -Wall -fopenmp omp-firstprivate.c -o omp-firstprivate


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>



int main() {
  
  int x = 22;

  #pragma omp parallel private(x)
  {
    printf("Thread %d: my private x=%d\n",omp_get_thread_num(),x);
  } // NOTE: implicit barrier sync here
  
  #pragma omp parallel firstprivate(x)
  {
    printf("Thread %d: my firstprivate x=%d\n",omp_get_thread_num(),x);
  } // NOTE: implicit barrier sync here


  return 0;
}
