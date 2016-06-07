#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

// this is the plain (no threads) version of dot product of two NxN matrices 
// compile with:
// gcc -O -Wall matmul-nothreads.c -o matmul-nothreads -DN=1024

// matrix dims N rows x N columns: use -DN=.. to define on compilation
//#define N 1024


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
int row,col,i;
double *pa,*pb,*pc,*parow;
double sum;
double ts,te;

// global variables for all threads
double *arbase;		// base of A matrix, in row major order
double *bcbase;		// base of B matrix, in column major order
double *crbase;		// base of C matirx, in row major order


  arbase = (double *)malloc(N*N*sizeof(double));
  if (arbase==NULL) {
    exit(1);
  }
  
  bcbase = (double *)malloc(N*N*sizeof(double));
  if (bcbase==NULL) {
    free(arbase); exit(1);
  }

  crbase = (double *)malloc(N*N*sizeof(double));
  if (crbase==NULL) {
    free(arbase); free(bcbase); exit(1);
  }

  // init input (and output) matrices
  for (i=0;i<N*N;i++) {
    arbase[i] = 2.0;
    bcbase[i] = 3.0;
    crbase[i] = 20.0;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  pc = crbase;	// pc = C[0][0]
  for (row=0;row<N;row++) {	// for each row of A
    parow = arbase + N*row;	// parow = A[row][0]  
    pb = bcbase;	// pb = B[0][0], for each A row loop
    for (col=0;col<N;col++) {	// for each column of B
      pa = parow;	// pa = A[row][0], for each B col loop    
      sum = 0.0;
      for (i=0;i<N;i++) {	// for each element of Arow,Bcol combination
        sum += *(pa)*(*pb);
        pa++; pb++;
      }
      *pc = sum;
      pc++;
      // here pb wraps around to next column
    }
    // here pc wraps around to next row
  }

  // get ending time
  get_walltime(&te);

  // print computation time
  printf("Computation time = %f sec\n",(te-ts));

  
  // test result
  for (i=0;i<N*N;i++) {
    if (crbase[i]!=6.0*N) { printf("Error! (%f)\n",crbase[i]); break; }  
  }

  free(crbase);
  free(bcbase);
  free(arbase);
  
  return 0;
}
