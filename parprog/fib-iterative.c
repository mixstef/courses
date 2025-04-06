// compile with: gcc -O2 -Wall -fopenmp fib-iterative.c -o fib-iterative

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#define N 13



int main() {
  int fib[N];

  #pragma omp parallel
  {
  
    #pragma omp single nowait
    {
    
      // this will be a task
      {
        printf("Computing fib(0) = ");
        fib[0] = 1;
        printf("%d\n",fib[0]);
      }
      
      // this will be a task
      {
        printf("Computing fib(1) = ");
        fib[1] = 1;
        printf("%d\n",fib[1]);
      }
   
      for (int i=2;i<N;i++) {
        // this will be a task
        {
          printf("Computing fib(%d) = ",i);        
          fib[i] = fib[i-1] + fib[i-2];
          printf("%d\n",fib[i]);
        }
      }
  
    }
  }
    
  return 0;
}
