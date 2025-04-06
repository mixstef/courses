

// Example usage of taskloop construct.

// Compile with: 
// gcc -O2 -Wall -fopenmp omp-taskloop-example.c -o omp-taskloop-example

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>


int main() {

  #pragma omp parallel
  {
  
    #pragma omp single nowait
    {
      
      #pragma omp taskloop
      for (int i=0;i<100;i++) {
          printf("Thread %d executing i =  %d\n",omp_get_thread_num(),i);
      }
      
      
      printf("Thread %d after task creation\n",omp_get_thread_num());

    }
  
  
  }

  return 0;
}
