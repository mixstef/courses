

// Example of for loop task generation with OpenMP.

// Compile with: 
// gcc -O2 -Wall -fopenmp omp-task-example2.c -o omp-task-example2

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>


int main() {

  #pragma omp parallel
  {
  
    #pragma omp single nowait
    {
      
      for (int i=0;i<10;i++) {
        #pragma omp task
        {
          printf("Thread %d executing task %c\n",omp_get_thread_num(),'A'+i);
        }
      }
      printf("Thread %d after thread creation\n",omp_get_thread_num());

    }
  
  
  }

  return 0;
}
