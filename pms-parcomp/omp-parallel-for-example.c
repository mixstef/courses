// Simple OpenMP demo of parallel for

// compile with:
// gcc -O2 -Wall -fopenmp omp-parallel-for-example.c -o omp-parallel-for-example


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#define N 100



int main() {
int i;

  #pragma omp parallel for private(i)
  for (i=0;i<N;i++)
  {
    // NOTE: i will be private even if not declared so
    
    printf("Thread %d working on element %d\n",omp_get_thread_num(),i);
        
  } 

  return 0;
}

