// compile with: gcc -fopenmp -O2 -Wall fib-tasks.c -o fib-tasks

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#define N 13


int fibonacci(int n) {
int i,j;

  printf("Thread %d computing fib(%d)\n",omp_get_thread_num(),n);

  if (n<2) return n;

  #pragma omp task shared(i)	// shared(i) NEEDED HERE, else firstprivate(i) by default
  i = fibonacci(n-1);
  
  #pragma omp task shared(j)	// shared(j) NEEDED HERE, else firstprivate(j) by default
  j = fibonacci(n-2);
  
  #pragma omp taskwait
  
  return i+j;
}


int main() {
int fib;

  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      fib = fibonacci(N);
    }  // no need for a barrier here
  }
  
  printf("fib(%d)= %d\n",N,fib);
  
  return 0;
}
