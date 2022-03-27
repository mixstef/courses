#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <pthread.h>

// this is the threaded version of dot product of two NxN matrices 
// compile with:
// gcc -Wall -O2 -pthread mmult-double-threads.c -o mmult-double-threads -DN=1000 -DTHREADS=4


// how many elements a thread will process - NOTE: surrounding ()!
#define BLOCKSIZE  ((N+THREADS-1)/THREADS)


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


// struct of info passed to each thread
struct thread_params {
  double *a;	// starting row of A for this thread
  double *b;	// base address of B (same for all threads)
  double *c;	// starting row of C for this thread
  int n;	// number of rows of A (and C) to process		
};


void *thread_func(void *args) {
 
  // get arguments
  struct thread_params *tp = (struct thread_params *)args;
  double *a = tp->a;
  double *b = tp->b;
  double *c = tp->c;
  int n = tp->n;
  
  // matrix multiplication (one "band" per thread)
  for (int i=0;i<n;i++) {	// for n rows, starting at a,c
  
    for (int j=0;j<N;j++) {	// for all "columns" (rows) of B
    
      double sum = 0.0;
      for (int k=0;k<N;k++) {	// for each element of selected A row and B "column"
        sum += a[i*N+k]*b[j*N+k];	// a[i,k]*b[j,k]  note: B is transposed, originally b[k,j]
      }
      c[i*N+j] = sum;	// c[i,j]
    
    }
  
  }
 

  // exit and let be joined
  pthread_exit(NULL);
}



int main() {
  pthread_t pid[THREADS];
  
  struct thread_params tparm[THREADS];



  double ts,te;

  double *a,*b,*c;	// matrices A,B,C C=AxB, B is transposed

  a = (double *)malloc(N*N*sizeof(double));
  if (a==NULL) {
    exit(1);
  }
  
  b = (double *)malloc(N*N*sizeof(double));
  if (b==NULL) {
    free(a); exit(1);
  }

  c = (double *)malloc(N*N*sizeof(double));
  if (c==NULL) {
    free(a); free(b); exit(1);
  }

  // init input and output matrices
  for (int i=0;i<N;i++) {
    for (int j=0;j<N;j++) {
      a[i*N+j] = i;
      b[i*N+j] = i;
      c[i*N+j] = 0.0;
    }
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  // for all threads
  for (int i=0;i<THREADS;i++) {

    // fill params for this thread
    tparm[i].a = a+i*N*BLOCKSIZE;
    tparm[i].b = b;
    tparm[i].c = c+i*N*BLOCKSIZE;
    
    if (i==(THREADS-1)) { // maybe less than blocksize to do...
      tparm[i].n = N-i*BLOCKSIZE; 
    }
    else { 
      tparm[i].n = BLOCKSIZE; // always blocksize work to do! 
    }
    
    // create i-th thread, pass ptr to tparm[i]
    if (pthread_create(&pid[i],NULL,thread_func,&tparm[i])!=0) {
      printf("Error in thread creation!\n");
      exit(1);
    }
  }

  // block until join
  for (int i=0;i<THREADS;i++) {
    if (pthread_join(pid[i],NULL)!=0) {
      printf("Error in thread join!\n");
      exit(1);  
    }
  }


  // get ending time
  get_walltime(&te);

  // print computation time
  printf("Computation time = %f sec\n",(te-ts));

  // test result
  int done = 0;
  for (int i=0;i<N&&done==0;i++) {
    for (int j=0;j<N;j++) {
      if (c[i*N+j]!=(double)i*j*N) { printf("Error! %d,%d (%f)\n",i,j,c[i*N+j]); done=1; break; }  
    }
  }

  free(c);
  free(b);
  free(a);
  
  return 0;
}
