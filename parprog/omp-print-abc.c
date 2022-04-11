// compile with: gcc -fopenmp -O2 -Wall omp-print-abc.c -o omp-print-abc

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>


int main() {

  #pragma omp parallel
  {
  
    #pragma omp single
    {
      
      printf("A\n");

      printf("B\n");

      printf("C\n");
      
    }
    
  }

  return 0;
}
