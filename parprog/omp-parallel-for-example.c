// Simple OpenMP demo of parallel for combined constructs

// compile with:
// gcc -O2 -Wall -fopenmp omp-parallel-for-example.c -o omp-parallel-for-example


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#define N 100



int main() {


  #pragma omp parallel for
  for (int i=0;i<N;i++)
  {
    
    printf("Thread %d working on element %d\n",omp_get_thread_num(),i);
        
  } 

  return 0;
}

