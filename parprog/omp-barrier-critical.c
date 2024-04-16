// Simple usage of barrier and critical region in OpenMP

// compile with:
// gcc -O2 -Wall -fopenmp omp-barrier-critical.c -o omp-barrier-critical


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>



int main() {

int id;		// normally this would be shared!
int total;	// shared
int count = 0;	// shared

  #pragma omp parallel private(id)
  {
    // NOTE: for id to be private, it could simply be declared in this block
    // or be in a function called in parallel block
    
    id = omp_get_thread_num();
    printf("Thread %d: Hello world!\n",id);
        
    // do something different by checking private thread id
    if (id==0) {
      printf("Thread 0: reading number of threads\n");
      total = omp_get_num_threads();
    }
    
    // ensure that thread 0 arrives here before any thread can continue
    #pragma omp barrier
    
    // all threads - update shared variable
    #pragma omp critical
    {
      count++;
      printf("Thread %d of %d: incr shared count to %d\n",id,total,count);
    }
    
  } 


  return 0;
}

