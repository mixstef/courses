// compile with: gcc -fopenmp -O2 -Wall omp-print-abc-taskwait.c -o omp-print-abc-taskwait

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

      #pragma omp taskwait	// suspend until children completed 
      
      printf("C\n");
      
    }
    
  }

  return 0;
}
