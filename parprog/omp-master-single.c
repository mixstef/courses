// Simple usage of master and single constructs in OpenMP

// compile with:
// gcc -O2 -Wall -fopenmp omp-master-single.c -o omp-master-single


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>



int main() {

  int total; // shared
  #pragma omp parallel
  {
    
    int id = omp_get_thread_num();
    printf("Thread %d: Hello world!\n",id);
        
    // master thread: do something different
    #pragma omp master
    {
      printf("Thread 0: reading number of threads\n");
      total = omp_get_num_threads();
    }
    
    // ensure that master thread arrives here before any thread can continue
    #pragma omp barrier
    
    // a randomly chosen thread only
    #pragma omp single
    {
      printf("Thread %d: number of threads is %d\n",id,total);
    }
    
  } 


  return 0;
}

