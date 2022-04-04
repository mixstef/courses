// Example showing the usage of a plain parallel construct without
// a worksharing construct: all threads will execute the same code!

// Compile with: gcc -O2 -Wall -fopenmp hello-omp.c -o hello-omp


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>



int main() {
  
  #pragma omp parallel
  {
    printf("Thread %d: Hello world!\n",omp_get_thread_num());
  
  } // NOTE: implicit barrier sync here

  return 0;
}
