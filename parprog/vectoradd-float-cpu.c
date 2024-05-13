// Sample vector addition performed on CPU.
// Compile with: gcc -Wall -O2 vectoradd-float-cpu.c -o vectoradd-float-cpu -DN=10000000

#include <stdio.h>
#include <stdlib.h>


int main() {
float *a,*b,*c;

  // allocate test arrays
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) { printf("Allocation failed!\n"); exit(1); }
  b = (float *)malloc(N*sizeof(float));
  if (b==NULL) { printf("Allocation failed!\n"); free(a); exit(1); }
  c = (float *)malloc(N*sizeof(float));
  if (c==NULL) { printf("Allocation failed!\n"); free(a); free(b); exit(1); }
  
  //initialize all arrays - cache warm-up
  for (int i=0;i<N;i++) {
    a[i]=2.0*i;
    b[i]=-i;
    c[i]=i+5.0;
  }
   
  // do artificial work
  for (int i=0;i<N;i++) {
    c[i] = a[i]+b[i];
  }
 
  
  // check result - avoid loop removal by compiler
  for (int i=0;i<N;i++) {
    if (c[i]!=a[i]+b[i]) {
      printf("Error!\n");
      break;
    }
  }
 
  
  // free arrays
  free(a); free(b); free(c);
  
  return 0;
}
