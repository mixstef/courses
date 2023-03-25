// Sample program generating THREADS identical threads with parametric args
// compile with e.g.: gcc -O2 -Wall -pthread example-no-barrier.c -o example-no-barrier -DTHREADS=10

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>


// struct of info passed to each thread
struct thread_params {
  int id;		// thread's id
};


void *thread_func(void *args) {
 
  // get arguments
  struct thread_params *tp = (struct thread_params *)args;
  int id = tp->id;
 
  // useful work A here - all threads
  printf("Thread %d: Work A\n",id);

  // useful work B here - thread 0 only
  if (id==0) {
    printf("Thread %d: Work B\n",id);    
  }

  // useful work C here - all threads
  printf("Thread %d: Work C\n",id);

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

  // block until join
  for (int i=0;i<THREADS;i++) {
    if (pthread_join(pid[i],NULL)!=0) {
      printf("Error in thread join!\n");
      exit(1);  
    }
  }
  
  return 0;

}
