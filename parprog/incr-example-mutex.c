// Increment of a shared variable guarded by mutex, *good* example.
// Compile with: gcc -O2 -Wall -pthread incr-example-mutex.c -o incr-example-mutex

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>
#include <unistd.h>	// for sleep()

#define THREADS 4
#define COUNT_MAX 20

// struct of info passed to each thread
struct thread_params {
  int id;
};


// global count variable
int global_count = 0;

// mutex protecting count variable
pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;


// sample thread function
void *thread_func(void *args) {

  struct thread_params *tp = (struct thread_params*)args;
  int id = tp->id;
  
  int done = 0;
  do {
  
    // lock count mutex
    pthread_mutex_lock(&count_mutex);
  
    printf("Thread %d: got count %d",id,global_count);
    if (global_count>=COUNT_MAX) {
      done = 1;
      printf(", terminating.\n");
    }
    else {
      global_count++;
      printf(", incrementing.\n");     
    }

    // unlock count mutex
    pthread_mutex_unlock(&count_mutex);

    // simulate some work
    sleep(1);
    
  } while (done==0);

  // exit and let be joined
  pthread_exit(NULL); 
}

int main() {
  
  pthread_t pid[THREADS];
  
  struct thread_params tparm[THREADS];

  // create threads
  for (int i=0;i<THREADS;i++) {
    // fill i-th member of tparm array
    tparm[i].id = i;
    
    // create i-th thread, pass ptr to tparm[i]
    if (pthread_create(&pid[i],NULL,thread_func,&tparm[i])!=0) {
      printf("Error in thread creation!\n");
      exit(1);
    }
  }

  // join threads
  for (int i=0;i<THREADS;i++) {
    if (pthread_join(pid[i],NULL)!=0) {
      printf("Error in thread join!\n");
      exit(1);  
    }
  }

  // destroy mutex - should be unlocked
  pthread_mutex_destroy(&count_mutex);

  return 0;
}
