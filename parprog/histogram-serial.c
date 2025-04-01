// Serial histogram example.
// Compile with: gcc -O2 -Wall histogram-serial.c -o histogram-serial -DN=10000000

// Adapted from:  https://ppc.cs.aalto.fi/ch3/memory/ 

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int collatz(long long x) {
  int i = 0;
  while (x != 1) {
    if (x % 2) {
      x = 3 * x + 1;
    }
    else {
      x = x/2;
    }
    i++;
  }  
  return i;
}


int main() {
  
  double ts,te;
  
  // histogram results and check arrays
  int result[200] = {0};
  int check[200] = {0};
  
  // allocate input array
  int *a = (int *)malloc(N*sizeof(int));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  

  // init array to Collatz step values 
  for (int i=1;i<=N;i++) {
    int ctz = collatz(i);
    if (ctz<200) check[ctz]++;
  }
 
  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // histogram computation
  for (int i=1;i<=N;i++) {
    int ctz = collatz(i);
    if (ctz<200) result[ctz]++;
  }
  
  // get ending time
  get_walltime(&te);

  // check result
  for (int i=0;i<200;i++) {
    if (result[i]!=check[i]) {
      printf("Histogram error!\n");
      break;
    }
  }

  // free array
  free(a);
 
  printf("Exec Time (sec) = %f\n",te-ts);

  
  return 0;
}

