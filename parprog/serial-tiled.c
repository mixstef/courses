// assignment2 - serial code, tiled version
// compile with:  gcc -Wall -O2 -fopenmp serial-tiled.c -o serial-tiled -DN=4000 -DTS=200


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

double *a;

  a = (double *)malloc(N*N*sizeof(double));
  if (a==NULL) {
    exit(1);
  }
  
  // init input elements of array
  for (int i=0;i<N;i++) {
    a[i] = 1.0;		// a[0,i]
    a[i*N] = 1.0;	// a[i,0]
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  // workload computation
  for (int ii=1;ii<N;ii+=TS) {		// starting row of tile
    int istart = ii;
    int iend = (istart+TS>N)?N:istart+TS;
  
    for (int jj=1;jj<N;jj+=TS) {	// starting column of tile
      int jstart = jj;
      int jend = (jstart+TS>N)?N:jstart+TS;
    
      for (int i=istart;i<iend;i++) {
        
        for (int j=jstart;j<jend;j++) {
    
          a[i*N+j] = a[(i-1)*N+j-1]// a[i,j] = a[i-1,j-1] + a[i-1][j] + a[i][j-1]
                +a[(i-1)*N+j]
                +a[i*N+j-1];
        }
      }     
    }
  
  }

  // get ending time
  get_walltime(&te);

  // print computation time
  printf("Computation time = %f sec\n",(te-ts));

  // test result
  int done = 0;
  for (int i=1;i<N && done==0;i++) {
    for (int j=0;j<N;j++) {
      if (i==0 || j==0) {
        if (a[i*N+j]!=1.0) {
          printf("Calculation error!\n");
          done = 1;
          break;
        }      
      }
      else if (a[i*N+j]!= a[(i-1)*N+j-1] + a[(i-1)*N+j] + a[i*N+j-1]) {
        printf("Calculation error!!\n");
        done = 1;
        break;
      }
    }
  }
  free(a);
  
  return 0;
}
