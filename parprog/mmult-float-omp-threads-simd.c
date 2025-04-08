// Float version of dot product of two NxN matrices 
// compile with: gcc -mavx -Wall -O2 -fopenmp mmult-float-omp-threads-simd.c -o mmult-float-omp-threads-simd -DN=1000

// NOTE: in order to be cache friendly, matrix B is assumed to be transposed


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

float *a,*b,*c;	// matrices A,B,C C=AxB, B is transposed

  int i = posix_memalign((void **)&a,32,N*N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); exit(1); }
  i = posix_memalign((void **)&b,32,N*N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); free(a); exit(1); }  
  i = posix_memalign((void **)&c,32,N*N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); free(a); exit(1); }
  
  // init input and output matrices
  for (int i=0;i<N*N;i++) {
    a[i] = rand()%10+1;
    b[i] = rand()%10+1;
    c[i] = 0.0;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  // load, matrix multiplication
  #pragma omp parallel for
  for (int i=0;i<N;i++) {	// for all rows of A,C
  
    for (int j=0;j<N;j++) {	// for all "columns" (rows) of B
    
      float sum = 0.0;
      #pragma omp simd reduction(+:sum) aligned(a,b,c:32)
      for (int k=0;k<N;k++) {	// for each element of selected A row and B "column"
        sum += a[i*N+k]*b[j*N+k];	// a[i,k]*b[j,k]  NOTE: B is transposed, originally b[k,j]
      }
      c[i*N+j] = sum;	// c[i,j]
    
    }
  
  }

  // get ending time
  get_walltime(&te);

  // print computation time
  printf("Computation time = %f sec\n",(te-ts));

  // check that all elements of c were "touched"
  for (int i=0;i<N*N;i++) {
    if (c[i]==0.0) { printf("Error!\n"); break; }
  }


  free(c);
  free(b);
  free(a);
  
  return 0;
}
