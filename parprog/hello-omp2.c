// Example showing the usage of a plain parallel construct without
// a worksharing construct: all threads will execute the same code!

// Compile with: gcc -O2 -Wall -fopenmp hello-omp2.c -o hello-omp2


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>



int main() {
  
  #pragma omp parallel
  {
    printf("Thread %d of %d: Hello world!\n",omp_get_thread_num(), omp_get_num_threads());
  
  } // NOTE: implicit barrier sync here

  omp_set_num_threads(3);
  
  #pragma omp parallel num_threads(16)
  {
    printf("Thread %d of %d: Nice world!\n",omp_get_thread_num(), omp_get_num_threads());
  
  } // NOTE: implicit barrier sync here


  #pragma omp parallel
  {
    printf("Thread %d of %d: Bye world!\n",omp_get_thread_num(), omp_get_num_threads());
  
  } // NOTE: implicit barrier sync here

  return 0;
}
