// Threaded prefix sum example, using the 3 phase algorithm (reduce, exclusive scan single, inclusive scan).
// Compile with: gcc -O2 -Wall -pthread prefix-sum-double-threads.c -o prefix-sum-double-threads -DN=10000000 -DTHREADS=4

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <pthread.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


// how many elements a thread will process - NOTE: surrounding ()!
#define BLOCKSIZE  ((N+THREADS-1)/THREADS)

// thread syncing barrier
pthread_barrier_t barrier;

// struct of info passed to each thread
struct thread_params {
  int id;		// thread's id (for demo purposes)
  double *a;	// ptr to thread's block start
  int n;	// how many elements to process
  double *r;	// start of partials' array
};


void *thread_func(void *args) {
 
  // get arguments
  struct thread_params *tp = (struct thread_params *)args;
  int id = tp->id;
  double *a = tp->a;
  int n = tp->n;
  double *r = tp->r;
 
  // step 1: reduce block - all threads except last one (with id=THREADS-1)
  if (id<THREADS-1) {
   double sum = 0.0;
   for (int i=0;i<n;i++) {
     sum += a[i];
   }
   r[id+1] = sum;	// store partial result, shifted 1 place to the right
  }

  // sync on barrier, for all threads
  pthread_barrier_wait(&barrier); // after sync, barrier goes to its init() state

  // step 2: (exclusive) prefix sum of partial results - single thread only (with id=0)
  if (id==0) {
    r[0] = 0.0;
    
    double sum = 0.0;
    for (int i=1;i<THREADS;i++) {
      sum += r[i];
      r[i] = sum;
    }    
  }

  // sync on barrier, for all threads
  pthread_barrier_wait(&barrier); // after sync, barrier goes to its init() state

  // step 3: (inclusive) prefix sum of block, ident is r[i] - all threads
  double sum = r[id];
  for (int i=0;i<n;i++) {
    sum += a[i];
    a[i] = sum;
  }

  // exit and let be joined
  pthread_exit(NULL);
}


int main() {
  
  double ts,te;

  pthread_t pid[THREADS];
  
  struct thread_params tparm[THREADS];
  
  double partial_sums[THREADS];

  // initialize barrier - always on all threads
  pthread_barrier_init(&barrier,NULL,THREADS);

  // allocate array
  double *a = (double *)malloc(N*sizeof(double));
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }	  

  // init array to 1..N
  for (int i=0;i<N;i++) {
    a[i] = i+1;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // for all threads
  for (int i=0;i<THREADS;i++) {
    // fill i-th member of tparm array
    tparm[i].id = i;
    tparm[i].a = a+i*BLOCKSIZE;
    tparm[i].r = partial_sums;	// NOTE: always &partial_sums[0]    

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


  // block until join
  for (int i=0;i<THREADS;i++) {
    if (pthread_join(pid[i],NULL)!=0) {
      printf("Error in thread join!\n");
      exit(1);  
    }
  }

  // get ending time
  get_walltime(&te);

  // check result
  for (int i=0;i<N;i++) {
    if (a[i]!=((double)(i+1)*(i+2))/2) {
      printf("Prefix sum error!\n");
      break;
    }
  }

  // free array
  free(a);

  // destroy barrier - no thread should be waiting on it
  pthread_barrier_destroy(&barrier);
 
  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
