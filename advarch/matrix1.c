#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// compile like:  gcc -Wall -O2 -DNROWS=10000 matrix1.c -o matrix1

// to be defined externaly with -D
//#define NROWS 1000

#define NCOLS 32

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
unsigned int i,j;
double sum;
double *table;
double ts,te,aps;


  table = (double *)malloc(NROWS*NCOLS*sizeof(double)); 
  if (table==NULL) {
    printf("alloc error!\n");
    exit(1);
  }

  // warmup
  for (i=0;i<NROWS*NCOLS;i++) {
     table[i] = 1.0;
  } 

  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // workload
  sum = 0.0;
  for (i=0;i<NROWS;i++) {
    for (j=0;j<NCOLS;j++) {
      sum += table[i*NCOLS+j];
    }
  }



  // get ending time
  get_walltime(&te);

  // DO NOT remove this: the compiler will optimize by removing test loops!
  printf("sum = %f\n",sum);

  // compute avg array element accesses /sec (total NROWS*NCOLS element accesses)
  aps = ((double)NROWS*NCOLS)/((te-ts)*1e6);
  
  printf("avg array element Maccesses/sec = %f\n",aps);

  free(table);

  return 0;
}

