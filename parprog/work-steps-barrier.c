// Sample program generating THREADS identical threads with parametric args, using barrier to sync between phases
// compile with e.g.: gcc -O2 -Wall -pthread work-steps-barrier.c -o work-steps-barrier -DTHREADS=10

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>


// thread syncing barrier
pthread_barrier_t barrier;


// struct of info passed to each thread
struct thread_params {
  int id;		// thread's id (for demo purposes)
};


void *thread_func(void *args) {
 
  // get arguments
  struct thread_params *tp = (struct thread_params *)args;
  int id = tp->id;
 
  // useful work A here - all threads
  printf("Thread %d: Work A\n",id);

  // sync on barrier, for all threads
  pthread_barrier_wait(&barrier); // after sync, barrier goes to its init() state

  // useful work B here - thread 0 only
  if (id==0) {
    printf("Thread %d: Work B\n",id);    
  }

  // sync on barrier, for all threads
  pthread_barrier_wait(&barrier); // after sync, barrier goes to its init() state

  // useful work C here - all threads
  printf("Thread %d: Work C\n",id);

  // exit and let be joined
  pthread_exit(NULL);
}



int main() {

  pthread_t pid[THREADS];
  
  struct thread_params tparm[THREADS];

  // initialize barrier - always on all threads
  pthread_barrier_init(&barrier,NULL,THREADS);
  
  // for all threads
  for (int i=0;i<THREADS;i++) {
    // fill i-th member of tparm array
    tparm[i].id = i;
    
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
  
  // destroy barrier - no thread should be waiting on it
  pthread_barrier_destroy(&barrier);
  
  return 0;

}
