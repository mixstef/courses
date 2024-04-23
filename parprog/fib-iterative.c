// compile with: gcc -O2 -Wall fib-iterative.c -o fib-iterative

#include <stdio.h>
#include <stdlib.h>


#define N 13



int main() {
int fib[N];

  fib[0] = 1;
  fib[1] = 1;
  
  for (int i=2;i<N;i++) {
    fib[i] = fib[i-1] + fib[i-2];
  }
  
  for (int i=0;i<N;i++) {
    printf("fib(%d)= %d\n",i,fib[i]);
  }
  
  return 0;
}
