// OpenMP prefix sum example.
// Compile with: gcc -O2 -Wall -fopenmp prefix-sum-double-omp.c -o prefix-sum-double-omp -DN=10000000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <omp.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
  
  double ts,te;

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

  // get max number of threads possible in parallel region
  int max_threads = omp_get_max_threads();
  
  // allocate array for partial results (shared)
  double *partial = (double *)malloc(max_threads*sizeof(double));
  if (a==NULL) {
    printf("Partials' allocation failed!\n");
    free(a); exit(1);
  } 
  
  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  #pragma omp parallel
  {
    int id = omp_get_thread_num();
  
    // step 1: reduce per thread block
    double sum = 0.0;
    #pragma omp for nowait
    for (int i=0;i<N;i++) {
      if (id<(max_threads-1))
        sum += a[i];
    }
    if (id<(max_threads-1)) {
      partial[id+1] = sum;
    }
    
    #pragma omp barrier
    
    // step 2: prefix sum of partials
    #pragma omp single
    {
      partial[0] = 0.0;
      double sum = 0.0;
      for (int i=1;i<omp_get_num_threads();i++) {
        sum += partial[i];
        partial[i] = sum;      
      }
    } // implicit barrier here
    
    // step 3: prefix sums per block, adding partials
    sum = partial[id];
    #pragma omp for nowait
    for (int i=0;i<N;i++) {
        sum += a[i];
        a[i] = sum;
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

  // free arrays
  free(partial);
  free(a);
 
  printf("Exec Time (sec) = %f\n",te-ts);

  
  return 0;
}
