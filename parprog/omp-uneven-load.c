// Simple OpenMP demo of parallel for combined constructs
// with a simulated uneven load per iteration

// compile with:
// gcc -O2 -Wall -fopenmp omp-uneven-load.c -o omp-uneven-load


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>	// for usleep()

#include <omp.h>

#define N 20



int main() {


  #pragma omp parallel for num_threads(4)
  for (int i=1;i<=N;i++)
  {
    
    printf("Thread %d working on element %d\n",omp_get_thread_num(),i);
    
    // simulate uneven load
    usleep(500*i);
        
  } 

  return 0;
}

