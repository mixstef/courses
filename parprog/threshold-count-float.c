// Scans all elements of a float array and counts values greater than a threshold

// compile with: gcc -Wall -O2 threshold-count-float.c -o threshold-count-float -DN=10000 -DR=10000 -DTHRESHOLD=50.0



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
  float *a,count;
  
  // 1. allocate array
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }
  
  // 2. init a array to random int values, from 0 to 99 (represented exactly as floats)
  float check = 0.0;
  for (int i=0;i<N;i++) {
    a[i] = rand()%100;
    if (a[i]>THRESHOLD) {
      check++;
    }
  }


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // 3. scan array (R times)
  for (int j=0;j<R;j++) {
    count = 0;
    for (int i=0;i<N;i++) {
      if (a[i]>THRESHOLD) {
        count++;
      }
    }
  }
  
  // get ending time
  get_walltime(&te);
  
  // 4. check result
  if (count!=check) {
    printf("Error!\n");
  }
  
  // 5. free arrays
  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
