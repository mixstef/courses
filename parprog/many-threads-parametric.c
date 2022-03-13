// Sample program generating THREADS identical threads with parametric args
// compile with e.g.:
// gcc -O2 -Wall -pthread many-threads-parametric.c -o many-threads-parametric -DTHREADS=10

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>


// struct of info passed to each thread
struct thread_params {
  int id;		// thread's id (for demo purposes)
};


void *thread_func(void *args) {
 
  // get arguments
  struct thread_params *tp = (struct thread_params *)args;
  int id = tp->id;
 
  // useful work here
  printf("Child thread %d working..\n",id);

  // exit and let be joined
  pthread_exit(NULL);
}



int main() {

  pthread_t pid[THREADS];
  
  struct thread_params tparm[THREADS];
  
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

  // useful work here
  printf("Main thread working..\n");

  // block until join
  for (int i=0;i<THREADS;i++) {
    if (pthread_join(pid[i],NULL)!=0) {
      printf("Error in thread join!\n");
      exit(1);  
    }
  }
  
  return 0;

}
