// compile with: gcc -fopenmp -O2 -Wall omp-print-abc-tasks.c -o omp-print-abc-tasks

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>


int main() {

  #pragma omp parallel
  {
  
    #pragma omp single
    {
      
      #pragma omp task
      printf("A\n");

      #pragma omp task
      printf("B\n");

      printf("C\n");
      
    }
    
  }

  return 0;
}
