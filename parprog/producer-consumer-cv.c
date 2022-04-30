// Producer-consumer example with condition variables.
// Compile with: gcc -O2 -Wall -pthread producer-consumer-cv.c -o producer-consumer-cv

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#define MESSAGES 20


// global integer buffer
int global_buffer;
// global avail messages count (0 or 1)
int global_availmsg = 0;	// empty

// condition variable, signals a put operation (receiver waits on this)
pthread_cond_t msg_in = PTHREAD_COND_INITIALIZER;
// condition variable, signals a get operation (sender waits on this)
pthread_cond_t msg_out = PTHREAD_COND_INITIALIZER;

// mutex protecting common resources
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


// producer thread function
void *producer_thread(void *args) {
  int i;
  
  // send a predefined number of messages
  for (i=0;i<MESSAGES;i++) {

    // lock mutex
    pthread_mutex_lock(&mutex);
    //printf("Producer: testing...\n");    
    while (global_availmsg>0) {	// NOTE: we use while instead of if! more than one thread may wake up
    				// cf. 'mesa' vs 'hoare' semantics
      pthread_cond_wait(&msg_out,&mutex);  // wait until a msg is received - NOTE: mutex MUST be locked here.
      					   // If thread is going to wait, mutex is unlocked automatically.
      					   // When we wake up, mutex will be locked by us again. 
      //printf("Producer: testing...\n");
    }
    // send message
    printf("Producer: sending msg %d\n",i);
    global_buffer = i;
    global_availmsg = 1;
    
    // signal the receiver that something was put in buffer
    pthread_cond_signal(&msg_in);
    
    // unlock mutex
    pthread_mutex_unlock(&mutex);
  }
  
  // exit and let be joined
  pthread_exit(NULL); 
}
  
  
// receiver thread function
void *consumer_thread(void *args) {
  int i;
  
  // receive a predefined number of messages
  for (i=0;i<MESSAGES;i++) {
    // lock mutex
    pthread_mutex_lock(&mutex);
    //printf("Consumer: testing...\n");
    while (global_availmsg<1) {	// NOTE: we use while instead of if! see above in producer code
      pthread_cond_wait(&msg_in,&mutex);
      
      //printf("Consumer: testing...\n"); 
    }
    
    // receive message
    printf("Consumer: received msg %d\n",global_buffer);
    global_availmsg = 0;
    
    // signal the sender that something was removed from buffer
    pthread_cond_signal(&msg_out);
    
    // unlock mutex
    pthread_mutex_unlock(&mutex);
  }
  
  // exit and let be joined
  pthread_exit(NULL); 
}


int main() {
  
  pthread_t producer,consumer;
  
  // create threads
  pthread_create(&producer,NULL,producer_thread,NULL);
  pthread_create(&consumer,NULL,consumer_thread,NULL);
  
  // then join threads
  pthread_join(producer,NULL);
  pthread_join(consumer,NULL);

  // destroy mutex - should be unlocked
  pthread_mutex_destroy(&mutex);

  // destroy cvs - no process should be waiting on these
  pthread_cond_destroy(&msg_out);
  pthread_cond_destroy(&msg_in);

  return 0;
}
