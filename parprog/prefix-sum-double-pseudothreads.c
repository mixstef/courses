// "Single-threaded" prefix sum example, using the 3 phase algorithm (reduce, exclusive scan single, inclusive scan) - preparation for mutithreading
// Compile with: gcc -O2 -Wall prefix-sum-double-pseudothreads.c -o prefix-sum-double-pseudothreads -DN=10000000 -DTHREADS=4

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


// how many elements a "thread" will process - NOTE: surrounding ()!
#define BLOCKSIZE  ((N+THREADS-1)/THREADS)


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
  
  double ts,te;

  double partial_sums[THREADS];

  // allocate array
  double *a = (double *)malloc(N*sizeof(double));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  

  // init array to 1..N
  for (int i=0;i<N;i++) {
    a[i] = i+1;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // step 1: reduce block - all "threads" except last one (with id=THREADS-1)
  for (int id=0;id<THREADS-1;id++) {
    // following will be thread's params
    double *pa = a+id*BLOCKSIZE;
    int n = (id==(THREADS-1))?N-id*BLOCKSIZE:BLOCKSIZE;
        
    double sum = 0.0;
    for (int i=0;i<n;i++) {
      sum += pa[i];
    }
    partial_sums[id+1] = sum;	// store partial result, shifted 1 place to the right
  }
  
  // step 2: (exclusive) prefix sum of partial results - single "thread" only
  partial_sums[0] = 0.0;
    
  double sum = 0.0;
  for (int i=1;i<THREADS;i++) {
    sum += partial_sums[i];
    partial_sums[i] = sum;
  }    

  // step 3: (inclusive) prefix sum of block, ident is r[i] - all "threads"
  for (int id=0;id<THREADS;id++) {
    // following will be thread's params
    double *pa = a+id*BLOCKSIZE;
    int n = (id==(THREADS-1))?N-id*BLOCKSIZE:BLOCKSIZE;
  
    double sum = partial_sums[id];
    for (int i=0;i<n;i++) {
      sum += pa[i];
      pa[i] = sum;
    }
  }
  
  // get ending time
  get_walltime(&te);

  // check result
  for (int i=0;i<N;i++) {
    if (a[i]!=((double)(i+1)*(i+2))/2) {
      printf("Prefix sum error!\n");
      break;
    }
  }

  // free array
  free(a);
 
  printf("Exec Time (sec) = %f\n",te-ts);

  
  return 0;
}
