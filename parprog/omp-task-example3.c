// Example of task generation with OpenMP.

// Compile with: 
// gcc -O2 -Wall -fopenmp omp-task-example3.c -o omp-task-example3

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>


void atasks() {
  #pragma omp task
  {
    printf("Thread %d executing task A1\n",omp_get_thread_num());
  }

  #pragma omp task
  {
    printf("Thread %d executing task A2\n",omp_get_thread_num());
  }
}

void btasks() {

  for (int i=0;i<10;i++) {
    #pragma omp task
    {
      printf("Thread %d executing task B%d\n",omp_get_thread_num(),i);
    }
  }

}


int main() {

  #pragma omp parallel
  {
  
    #pragma omp single nowait
    {
      
      #pragma omp task
      {
        atasks();
      }

      #pragma omp task
      {
        btasks();
      }
      
      printf("Thread %d after thread creation\n",omp_get_thread_num());
    }
  
  
  }

  return 0;
}
