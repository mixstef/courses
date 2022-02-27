#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// compile with: gcc -Wall -O2 filter-int.c -o filter-int -DN=10000000 -DR=10


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}



int main() {
  int *a,*b;
  double ts,te;
    
  a = (int *)malloc(N*sizeof(int));
  if (a==NULL) { printf("alloc error\n"); exit(1); }
  b = (int *)malloc(N*sizeof(int)); 
  if (b==NULL) { free(a); printf("alloc error\n"); exit(1); }
  
  for (int i=0;i<N;i++) {
    a[i] = i % 100;
    b[i] = -2*i;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  for (int j=0;j<R;j++) {  
    for (int i=0;i<N;i++) {
      if (a[i]>50) {
        b[i] = 1;
      }
      else {
        b[i] = 0;
      }     
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


