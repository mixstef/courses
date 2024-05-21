// Sum reduction of an array of N floats, CPU version

// Compile with: gcc -Wall -O2 sum-reduction-float-cpu.c -o sum-reduction-float-cpu -DN=10000000


#include <stdio.h>
#include <stdlib.h>



int main() {
  float *a,sum;
  
  // allocate input array
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  
  
  // init array to random int values, from 0 to 2 (avoid float truncation errors)
  for (int i=0;i<N;i++) {
    a[i] = rand()%3;
  }
  
  // reduce array to sum
  sum = 0.0;
  for (int i=0;i<N;i++) {
    sum += a[i];
  }
  
  // check result (dummy code that always succeeds, sum computed as before)
  float check = 0.0;
  for (int i=0;i<N;i++) {
    check += a[i];
  }
  
  if (check!=sum) {
    printf("Error! found %f instead of %f\n",sum,check);
  }
  else {
    printf("Success, result = %f\n",sum);
  }
  
  free(a);
  
  return 0;
}
