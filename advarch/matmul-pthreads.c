#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include <pthread.h>

// this is the threaded (pthreads) version of dot product of two NxN matrices 
// compile with:
// gcc -pthread -O2 -Wall matmul-pthreads.c -o matmul-pthreads -DN=1024 -DTHREADS=2

// number of threads: use -DTHREADS=.. to define on compilation
//#define THREADS 2

// matrix dims N rows x N columns: use -DN=.. to define on compilation
//#define N 1024

// how many rows a thread will process - NOTE: surrounding ()!
#define BLOCKSIZE  ((N+THREADS-1)/THREADS)

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

// struct of info passed to each thread
struct thread_params {
  double *starta;
  double *startb;
  double *startc;
  int rownum;
};



// threaded function (should be void * returning by specs)
void *work(void *args) {
double *pa,*pb,*pc,*parow;
double sum;
int row,col,i,n;
  
  struct thread_params *t = (struct thread_params *)args;
  n = t->rownum;
    
  pc = t->startc;	// pc = C[chunkbegin][0]
  for (row=0;row<n;row++) {	// for each row of A in chunk
    parow = t->starta + N*row;	// parow = A[chunkbegin+row][0]  
    pb = t->startb;	// pb = B[0][0], for each A row loop
    for (col=0;col<N;col++) {	// for each column of B
      pa = parow;	// pa = A[chunkbegin+row][0], for each B col loop    
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

  // exit and let be joined
  pthread_exit(NULL); 

}


int main() {
int i;
double ts,te;


// thread ids (opaque handles) - used for join
pthread_t threadids[THREADS];	

// array of structs to fill and pass to threads on creation
struct thread_params tparm[THREADS];

// global variables for all threads
double *arbase;		// base of A matrix, in row major order
double *bcbase;		// base of B matrix, in column major order
double *crbase;		// base of C matirx, in row major order


  printf("Mult %dx%d with %d threads (%d rows per thread)\n",N,N,THREADS,BLOCKSIZE);

  
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

  // create worker threads
  for (i=0;i<THREADS;i++) {
    tparm[i].starta = arbase+i*BLOCKSIZE*N;
    tparm[i].startb = bcbase;
    tparm[i].startc = crbase+i*BLOCKSIZE*N;
    if (i==(THREADS-1)) {	// last block of table
      tparm[i].rownum = N - i*BLOCKSIZE;
    }
    else tparm[i].rownum = BLOCKSIZE;	// all other blocks
    
    pthread_create(&threadids[i],NULL,work,&tparm[i]);
  }
  // then join threads
  for (i=0;i<THREADS;i++) {
    pthread_join(threadids[i],NULL);
  }

  // get ending time
  get_walltime(&te);

  // print computation time
  printf("Computation time = %f sec\n",(te-ts));
  
  // test result
  for (i=0;i<N*N;i++) {
    if (crbase[i]!=6.0*N) { printf("Error! (%f) at index %d\n",crbase[i],i); break; }  
  }

  free(crbase);
  free(bcbase);
  free(arbase);
  
  return 0;
}
