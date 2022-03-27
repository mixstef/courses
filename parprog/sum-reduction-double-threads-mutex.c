// Threaded sum-reduction mutex example, threads sum blocks of input array and update global sum at the end.
// Compile with: gcc -O2 -Wall -pthread sum-reduction-double-threads-mutex.c -o sum-reduction-double-threads-mutex -DN=10000000 -DTHREADS=4

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


// struct of info passed to each thread
struct thread_params {
  double *a;	// start of block on input array
  int n;	// how many elements to sum
};


// global sum variable
double g_sum = 0.0;

// mutex protecting sum variable
pthread_mutex_t sum_mutex = PTHREAD_MUTEX_INITIALIZER;


// sample thread function
void *thread_func(void *args) {

  struct thread_params *tp = (struct thread_params*)args;
  double *a = tp->a;
  int n = tp->n;
  
  // compute local sum of block

  double mysum = 0.0;	// private to each thread
  for (int i=0;i<n;i++) {
    mysum += a[i];
  }
  
  // update global sum
  
  // lock sum mutex
  pthread_mutex_lock(&sum_mutex);
  
  g_sum += mysum;  
  
  // unlock sum mutex
  pthread_mutex_unlock(&sum_mutex);


  // exit and let be joined
  pthread_exit(NULL); 
}



int main() {
  
  double ts,te;
  
  pthread_t pid[THREADS];
  
  struct thread_params tparm[THREADS];
  
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
    tparm[i].a = a+i*BLOCKSIZE;    

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
  double result = ((double)N*(N+1))/2;  
  if (g_sum!=result) {
    printf("Reduction error!\n");
  }


  // destroy mutex - should be unlocked
  pthread_mutex_destroy(&sum_mutex);

  // free array
  free(a);
  
  printf("Exec Time (sec) = %f\n",te-ts);  
  
  return 0;
}
