// benchmark template: computing the sum of an 2d array of doubles

// compile with: gcc -Wall -O2 array2d-sum.c -o array2d-sum -DN=1000 -DR=100
// check assembly output: gcc -Wall -O2 array2d-sum.c -S -DN=1000 -DR=100


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
double ts,te,aps;
double *a;

  // 1. allocate array
  a = (double *)malloc(N*N*sizeof(double)); 
  if (a==NULL) {
    printf("alloc error!\n");
    exit(1);
  }

  // 2. init array to 1..N
  for (int i=0;i<N*N;i++) {
     a[i] = i+1;
  } 

  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // 3. Workload
  double sum = 0.0;
  for (int k=0;k<R;k++) {
    for (int i=0;i<N;i++) {
      for (int j=0;j<N;j++) {
        sum += a[i*N+j];
      }
    }
  }

  // get ending time
  get_walltime(&te);

  // 4. DO NOT remove this: the compiler will optimize by removing test loops!
  printf("sum = %f\n",sum);

  // compute avg array element accesses /sec (total R*N*N element accesses)
  aps = ((double)R*N*N)/((te-ts)*1e6);
  
  printf("avg array element Maccesses/sec = %f\n",aps);

  free(a);

  return 0;
}

