// Assignment2 OpenMP tasks code, 1 element per task (>20x slower than serial!)
// compile with:  gcc -Wall -O2 -fopenmp single-cell-task.c -o single-cell-task -DN=4000


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

  #pragma omp parallel
  {
  
    #pragma omp single
    {
      // workload computation
      for (int i=1;i<N;i++) {	// for all rows of A (except first)
  
        for (int j=1;j<N;j++) {	// for all columns or row (except first)
    
          #pragma omp task depend(out:a[i*N+j]) \
                           depend(in:a[(i-1)*N+j-1],a[(i-1)*N+j],a[i*N+j-1])
          {
            a[i*N+j] = a[(i-1)*N+j-1] // a[i,j] = a[i-1,j-1] + a[i-1][j] + a[i][j-1]
                      +a[(i-1)*N+j]
                      +a[i*N+j-1];
          }     
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
