// compile with: gcc -O2 -Wall -fopenmp fib-iterative-tasks.c -o fib-iterative-tasks

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
      #pragma omp task depend(out:fib[0])
      {
        printf("Thread %d: computing fib(0) = ",omp_get_thread_num());
        fib[0] = 1;
        printf("%d\n",fib[0]);
      }
  
      #pragma omp task depend(out:fib[1])
      {
        printf("Thread %d: computing fib(1) = ",omp_get_thread_num());
        fib[1] = 1;
        printf("%d\n",fib[1]);
      }
      
      for (int i=2;i<N;i++) {
        #pragma omp task depend(in:fib[i-1],fib[i-2]) depend(out:fib[i])
        {
          printf("Thread %d: computing fib(%d) = ",omp_get_thread_num(),i);        
          fib[i] = fib[i-1] + fib[i-2];
          printf("%d\n",fib[i]);
        }
      }
    }
  }
  
  
  return 0;
}
