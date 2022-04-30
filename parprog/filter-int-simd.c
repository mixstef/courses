// compile with: gcc -msse2 -Wall -O2 -fopenmp filter-int-simd.c -o filter-int-simd -DN=10000000 -DR=10


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
  int *a,*b;
  double ts,te;
    
  int t = posix_memalign((void **)&a,16,N*sizeof(int));  
  if (t!=0) { printf("alloc error\n"); exit(1); }
  t = posix_memalign((void **)&b,16,N*sizeof(int));  
  if (t!=0) { free(a); printf("alloc error\n"); exit(1); }
  
  for (int i=0;i<N;i++) {
    a[i] = i % 100;
    b[i] = -2*i;
  }
  
  // get starting time (double, seconds) 
  get_walltime(&ts);

  for (int j=0;j<R;j++) {
  
    #pragma omp simd aligned(a,b:16)  
    for (int i=0;i<N;i++) {
      b[i] = (a[i]>50)?1:0;
    }
  } 

  // get ending time
  get_walltime(&te);

  for (int i=0;i<N;i++) {
    if (((a[i]>50)&&(b[i]!=1))||((a[i]<=50)&&(b[i]!=0))) {
      printf("error\n");
      break;
    }
  }
  
  free(b);
  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);

  return 0;
}


