// Sample program generating THREADS identical threads with parametric args
// working on parts of an array of size N
// compile with e.g.:
// gcc -O2 -Wall -pthread blocksize-demo.c -o blocksize-demo -DN=100 -DTHREADS=10

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>


// how many elements a thread will process - NOTE: surrounding ()!
#define BLOCKSIZE  ((N+THREADS-1)/THREADS)


// struct of info passed to each thread
struct thread_params {
  int id;	// thread's id (for demo purposes)
  double *pa;	// start of array to work on	
  int n;	// how many items to "process"
};


void *thread_func(void *args) {
 
  // get arguments
  struct thread_params *tp = (struct thread_params *)args;
  int id = tp->id;
  int n = tp->n;
  double *pa = tp->pa;
 
  // useful work here
  for (int i=0;i<n;i++) {
    printf("Child thread %d working on element %f\n",id,pa[i]);
  }

  // exit and let be joined
  pthread_exit(NULL);
}



int main() {

  pthread_t pid[THREADS];
  
  struct thread_params tparm[THREADS];

  // allocate array
  double *a = (double *)malloc(N*sizeof(double));
  if (a==NULL) { printf("alloc error\n"); exit(1); }

  // init array
  for (int i=0;i<N;i++) {
    a[i] = i;
  }
  
  // for all threads
  for (int i=0;i<THREADS;i++) {
    // fill i-th member of tparm array
    tparm[i].id = i;
    tparm[i].pa = a+i*BLOCKSIZE;
    if (i==(THREADS-1)) { // last thread, maybe less than blocksize to do...
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

  // main thread continues after thread creation
  printf("Main thread, thread creation finished\n");

  // block until join
  for (int i=0;i<THREADS;i++) {
    if (pthread_join(pid[i],NULL)!=0) {
      printf("Error in thread join!\n");
      exit(1);  
    }
  }
  
  
  free(a);
  
  return 0;

}
