// Example of task generation with OpenMP.

// Compile with: 
// gcc -O2 -Wall -fopenmp omp-task-example1.c -o omp-task-example1

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>


int main() {

  #pragma omp parallel
  {
  
    #pragma omp single nowait
    {
      #pragma omp task
      {
        printf("Thread %d executing task A\n",omp_get_thread_num());
      }

      #pragma omp task
      {
        printf("Thread %d executing task B\n",omp_get_thread_num());
      }
      
      
      printf("Thread %d after thread creation\n",omp_get_thread_num());
    }
  
  
  }

  return 0;
}
