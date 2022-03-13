// Sample program generating a single thread
// compile with e.g.:
// gcc -O2 -Wall -pthread one-thread.c -o one-thread

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>


void *thread_func(void *args) {
 
  // useful work here
  printf("Child thread working..\n");

  // exit and let be joined
  pthread_exit(NULL);
}



int main() {

  pthread_t pid;
  
  // create a thread
  if (pthread_create(&pid,NULL,thread_func,NULL)!=0) {
    printf("Error in thread creation!\n");
    exit(1);
  }

  // useful work here
  printf("Main thread working..\n");


  // block until join
  if (pthread_join(pid,NULL)!=0) {
    printf("Error in thread join!\n");
    exit(1);  
  }
  
  return 0;

}
