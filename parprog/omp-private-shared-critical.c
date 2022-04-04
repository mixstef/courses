// Simple usage of private variables and critical region in OpenMP

// compile with:
// gcc -O2 -Wall -fopenmp omp-private-shared-critical.c -o omp-private-shared-critical


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>



int main() {

int id;		// normally this would be shared!
int count = 0;	// shared

  #pragma omp parallel private(id)
  {
    // NOTE: for id to be private, it could simply be declared in this block
    // or be in a function called in parallel block
    
    id = omp_get_thread_num();
    printf("Thread %d: Hello world!\n",id);
    
    
    // do something different by checking private thread id
    if (id==0) {
      printf("Thread %d: Total number of threads is %d!\n",id,omp_get_num_threads());
    }
    
    // all threads - update shared variable
    #pragma omp critical
    {
      count++;
      printf("Thread %d: incr shared count to %d\n",id,count);
    }
    
  } 


  return 0;
}

