// Simple OpenMP demo of worksharing for construct

// compile with:
// gcc -O2 -Wall -fopenmp omp-for-example.c -o omp-for-example


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#define N 100



int main() {

  #pragma omp parallel
  {
  
    #pragma omp for
    for (int i=0;i<N;i++) {    

      printf("Thread %d working on element %d\n",omp_get_thread_num(),i);
    
    } // implicit barrier here - use nowait clause to avoid!
    
      
  } // implicit barrier here 

  return 0;
}

