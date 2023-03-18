#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <emmintrin.h>

// this is the sse float version of dot product of two NxN matrices 
// compile with: gcc -msse2 -Wall -O2 mmult-float-sse.c -o mmult-float-sse -DN=1000

// N must be a multiple of 4!

// NOTE: in order to be cache friendly, matrix B is assumed to be transposed


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
double ts,te;

float *a,*b,*c;	// matrices A,B,C C=AxB, B is transposed
__m128 *pa,*pb;
__m128 sum,t1,t2;

  int i = posix_memalign((void **)&a,16,N*N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); exit(1); }
  i = posix_memalign((void **)&b,16,N*N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); free(a); exit(1); }
  i = posix_memalign((void **)&c,16,N*N*sizeof(float));
  if (i!=0) { printf("Allocation failed!\n"); free(b); free(a); exit(1); }

  // alias m128 ptrs to arrays
  pa = (__m128 *)a;
  pb = (__m128 *)b;

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
    
      sum = _mm_setzero_ps();
      for (int k=0;k<N/4;k++) {	// for each element of selected A row and B "column"
        sum = _mm_add_ps(sum,_mm_mul_ps(pa[i*N/4+k],pb[j*N/4+k])); // note: B is transposed
      }
      
      // perform horizontal sum
      t1 = _mm_shuffle_ps(sum,sum,_MM_SHUFFLE(2,3,0,1));	// C D A B
      t2 = _mm_add_ps(sum,t1);				// D+C C+D B+A A+B
      t1 = _mm_movehl_ps(t1,t2);				// C D D+C C+D
      t2 = _mm_add_ss(t1,t2);				// C D D+C A+B+C+D
      
      c[i*N+j] = _mm_cvtss_f32(t2);	// c[i,j]
    }
  
  }

  // get ending time
  get_walltime(&te);

  // print computation time
  printf("Computation time = %f sec\n",(te-ts));

  // test result (i.e. check that all elements of c were "touched"
  for (int i=0;i<N*N;i++) {
    if (c[i]==0.0) { printf("Error!\n"); break; }
  }


  free(c);
  free(b);
  free(a);
  
  return 0;
}
