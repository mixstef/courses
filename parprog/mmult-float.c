#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

// this is the float version of dot product of two NxN matrices 
// compile with: gcc -Wall -O2 mmult-float.c -o mmult-float -DN=1000

// matrix dims N rows x N columns: use -DN=.. to define on compilation



void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
double ts,te;

float *a,*b,*c;	// matrices A,B,C C=AxB, B is transposed

  a = (float *)malloc(N*N*sizeof(float));
  if (a==NULL) {
    exit(1);
  }
  
  b = (float *)malloc(N*N*sizeof(float));
  if (b==NULL) {
    free(a); exit(1);
  }

  c = (float *)malloc(N*N*sizeof(float));
  if (c==NULL) {
    free(a); free(b); exit(1);
  }

  // init input and output matrices
  for (int i=0;i<N*N;i++) {
    a[i] = rand()%10+1;
    b[i] = rand()%10+1;
    c[i] = 0.0;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  // load, matrix multiplication
  for (int i=0;i<N;i++) {	// for all rows of A,C
  
    for (int j=0;j<N;j++) {	// for all "columns" (rows) of B
    
      float sum = 0.0;
      for (int k=0;k<N;k++) {	// for each element of selected A row and B "column"
        sum += a[i*N+k]*b[j*N+k];	// a[i,k]*b[j,k]  note: B is transposed, originally b[k,j]
      }
      c[i*N+j] = sum;	// c[i,j]
    
    }
  
  }

  // get ending time
  get_walltime(&te);

  // print computation time
  printf("Computation time = %f sec\n",(te-ts));

  // test result
  for (int i=0;i<N*N;i++) {
    if (c[i]==0.0) { printf("Error!\n"); break; }
  }


  free(c);
  free(b);
  free(a);
  
  return 0;
}
