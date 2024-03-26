// AVX float version of dot product of two NxN matrices 
// compile with: gcc -mavx -Wall -O2 mmult-float-avx.c -o mmult-float-avx -DN=1000

// N must be a multiple of 4!

// NOTE: in order to be cache friendly, matrix B is assumed to be transposed


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <immintrin.h>



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
  c = (float *)malloc(N*N*sizeof(float));
  if (c==NULL) { printf("Allocation failed!\n"); free(a); free(b); exit(1); }
  
  // alias m256 ptrs to arrays
  __m256 *pa = (__m256 *)a;
  __m256 *pb = (__m256 *)b;

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
    
      __m256 sum = _mm256_setzero_ps();
      for (int k=0;k<N/8;k++) {	// for each 8-float element of selected A row and B "column"
        sum = _mm256_add_ps(sum,_mm256_mul_ps(pa[i*N/8+k],pb[j*N/8+k])); // note: B is transposed
      }
      
      // horizontal sum of the 8 packed floats
      // add high and low 128-bit halves into a __m128 number t1 = h+d g+c f+b e+a 
      __m128 t1 = _mm_add_ps(_mm256_extractf128_ps(sum,1),
                             _mm256_castps256_ps128(sum));
      // create t2, a shuffled version of t1,  t2 = g+c h+d e+a f+b
      __m128 t2 = _mm_shuffle_ps(t1,t1,_MM_SHUFFLE(2,3,0,1)); 
      // add t1 and t2, result t3 = h+d+g+c g+c+h+d f+b+e+a e+a+f+b    
      __m128 t3 = _mm_add_ps(t1,t2);
      // set t2 to upper parts of t2 and t3, now t2 = g+h h+d h+d+g+c g+c+h+d 
      t2 = _mm_movehl_ps(t2,t3);
      // add lowest parts of t2 and t3, now t3 = g+h h+d h+d+g+c g+c+h+d+e+a+f+b  
      t3 = _mm_add_ss(t2,t3);
      // copy to a single float
      c[i*N+j] = _mm_cvtss_f32(t2);	// c[i,j]
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
