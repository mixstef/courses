// sum reduction of an array of floats

// compile with: gcc -Wall -O2 sum-float.c -o sum-float -DN=1000 -DR=100000

// NOTE: float result has not the accuracy required by big values of N!


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}




int main() {
  double ts,te;
  float *a,sum;
  
  // 1. allocate array
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  
  
  // 2. init array to 1..N
  for (int i=0;i<N;i++) {
    a[i] = i+1;
  }


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // 3. reduce array to sum (R times)
  for (int j=0;j<R;j++) {
    sum = 0;
    for (int i=0;i<N;i++) {
      sum += a[i];
    }
  }
  
  // get ending time
  get_walltime(&te);
  
  // 4. print result (no check due to float rounding)
  printf("Result = %f\n",sum);

  // 5. free array
  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
