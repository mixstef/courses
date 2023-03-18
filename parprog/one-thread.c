// Sample program generating a single thread
// compile with:  gcc -O2 -Wall -pthread one-thread.c -o one-thread

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>


void *thread_func(void *args) {
 
  // useful work here
  printf("Child thread working..\n");

  // terminate and let be joined
  pthread_exit(NULL);
}



int main() {

  pthread_t pid;  // the thread's "handle"
  
  // create a thread - non-blocking call
  if (pthread_create(&pid,NULL,thread_func,NULL)!=0) {
    printf("Error in thread creation!\n");
    exit(1);
  }

  // useful work here
  printf("Main thread working..\n");


  // wait until thread terminates - blocking call
  if (pthread_join(pid,NULL)!=0) {
    printf("Error in thread join!\n");
    exit(1);  
  }
  
  return 0;

}
