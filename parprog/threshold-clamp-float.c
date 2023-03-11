// Scans all elements of a float array and clamps values to a threshold, producing a new array as output

// compile with: gcc -Wall -O2 threshold-clamp-float.c -o threshold-clamp-float -DN=10000 -DR=10000 -DTHRESHOLD=50.0

// NOTE: check what happens when THRESHOLD goes towards 0 or 99 (e.g. 10.0 or 90.0)!

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
  float *a,*b;
  
  // 1. allocate arrays
  a = (float *)malloc(N*sizeof(float));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }
  b = (float *)malloc(N*sizeof(float));
  if (b==NULL) {
    printf("Allocation failed!\n");
    free(a);
    exit(1);
  }  	  
  
  // 2. init a array to random int values, from 0 to 99 (represented exactly as floats)
  for (int i=0;i<N;i++) {
    a[i] = rand()%100;
    b[i] = -i;	// init b array to negative numbers
  }


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // 3. scan array (R times)
  for (int j=0;j<R;j++) {

    for (int i=0;i<N;i++) {
      if (a[i]>THRESHOLD) {
        b[i] = THRESHOLD;
      }
      else {
        b[i] = a[i];
      }
    }
  }
  
  // get ending time
  get_walltime(&te);
  
  // 4. check result
  for (int i=0;i<N;i++) {
    if ((a[i]<=THRESHOLD && b[i]!=a[i])||(a[i]>THRESHOLD && b[i]!=THRESHOLD)) {
      printf("Error!\n");
      break;
    }
  }
  
  // 5. free arrays
  free(a);
  free(b);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
