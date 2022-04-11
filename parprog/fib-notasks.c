// compile with: gcc -O2 -Wall fib-notasks.c -o fib-notasks

#include <stdio.h>
#include <stdlib.h>


#define N 13


int fibonacci(int n) {
int i,j;

  printf("Computing fib(%d)\n",n);
  
  if (n<2) return n;

  i = fibonacci(n-1);
  
  j = fibonacci(n-2);
  
  return i+j;
}


int main() {
int fib;

  fib = fibonacci(N);
  
  printf("fib(%d)= %d\n",N,fib);
  
  return 0;
}
