// ERRONEOUS producer/consumer example
// Compile with: gcc -O2 -Wall -pthread producer-consumer-BAD.c -o producer-consumer-BAD

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#define MESSAGES 20


// global integer buffer
int global_buffer;

// global avail messages count (0 or 1)
int global_availmsg = 0;	// empty


// producer thread function
void *producer_thread(void *args) {
  int i;
  
  // send a predefined number of messages
  for (i=0;i<MESSAGES;i++) {
    
    while (global_availmsg>0) {
      // while buffer full, wait...
    }
    
    // send message
    printf("Producer: sending msg %d\n",i);
    global_buffer = i;
    global_availmsg = 1;	// mark buffer as full
  
  }
      
  // exit and let be joined
  pthread_exit(NULL); 
}
  
  
// receiver thread function
void *consumer_thread(void *args) {
  int i;
  
  // receive a predefined number of messages
  for (i=0;i<MESSAGES;i++) {
  
    while (global_availmsg<1) {
      // while buffer empty, wait...
    }
  
    // receive message
    printf("Consumer: received msg %d\n",global_buffer);
    global_availmsg = 0;	// mark buffer as empty

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


  return 0;
}
