// Sum reduction of an array of N floats, repeated R times

// Compile with: gcc -Wall -O2 sum-float.c -o sum-float -DN=10000 -DR=10000



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
  
  // 2. init array to random int values, from 0 to 9 (represented exactly as floats)
  for (int i=0;i<N;i++) {
    a[i] = rand()%10;
  }


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // 3. reduce array to sum (R times)
  for (int j=0;j<R;j++) {
    sum = 0.0;
    for (int i=0;i<N;i++) {
      sum += a[i];
    }
  }
  
  // get ending time
  get_walltime(&te);
  
  // 4. check result (dummy code that always succeeds, sum computed as before)
  float check = 0.0;
  for (int i=0;i<N;i++) {
    check += a[i];
  }
  if (check!=sum) {
    printf("Error! found %f instead of %f\n",sum,check);
  }
  
  // 5. free array
  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
