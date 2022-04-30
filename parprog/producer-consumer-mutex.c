// Cycle burning producer/consumer example with mutex
// Compile with: gcc -O2 -Wall -pthread producer-consumer-mutex.c -o producer-consumer-mutex

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#define MESSAGES 20


// global integer buffer
int global_buffer;

// global avail messages count (0 or 1)
int global_availmsg = 0;	// empty

// mutex protecting common resources
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


// producer thread function
void *producer_thread(void *args) {
  int i;
  
  // send a predefined number of messages
  for (i=0;i<MESSAGES;i++) {

    pthread_mutex_lock(&mutex);
    //printf("Producer: testing...\n");    // add this to show wasted cycles    
    while (global_availmsg>0) {
      pthread_mutex_unlock(&mutex);

      // while buffer full, wait...

      pthread_mutex_lock(&mutex);              
      //printf("Producer: testing...\n");    // add this to show wasted cycles    
    }
    
    // send message
    printf("Producer: sending msg %d\n",i);
    global_buffer = i;
    global_availmsg = 1;	// mark buffer as full
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

    pthread_mutex_lock(&mutex);
    //printf("Consumer: testing...\n");    // add this to show wasted cycles
    while (global_availmsg<1) {
      pthread_mutex_unlock(&mutex);
        
      // while buffer empty, wait...
      
      pthread_mutex_lock(&mutex);
      //printf("Consumer: testing...\n");    // add this to show wasted cycles      
    }
  
    // receive message
    printf("Consumer: received msg %d\n",global_buffer);
    global_availmsg = 0;	// mark buffer as empty
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

  return 0;
}
