// Example of a queue implementation, global variables protected by mutex.
// Compile with: gcc -O2 -Wall -pthread queue-example-mutex.c -o queue-example-mutex

#include <stdio.h>
#include <stdlib.h>


#include <pthread.h>

#define MESSAGES 20

#define QUEUE_SIZE 7

// ---- globals ----

// global integer buffer
int global_buffer[QUEUE_SIZE];
int global_qin = 0;	// insertion index
int global_qout = 0;	// extraction index


// global avail messages count
int global_availmsg = 0;	// empty

// mutex protecting common resources
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


// ---- send/receive functions ----

void send_msg(int msg) {

    pthread_mutex_lock(&mutex);
    //printf("Producer: testing...\n");    // add this to show wasted cycles    
    while (global_availmsg>=QUEUE_SIZE) { 
      pthread_mutex_unlock(&mutex);

      // while buffer full, wait...

      pthread_mutex_lock(&mutex);              
      //printf("Producer: testing...\n");    // add this to show wasted cycles    
    }
    
    // send message
    global_buffer[global_qin] = msg;
    global_qin += 1;
    if (global_qin>=QUEUE_SIZE) global_qin = 0; // wrap around
    global_availmsg += 1;

    pthread_mutex_unlock(&mutex);      
}


int recv_msg() {

    pthread_mutex_lock(&mutex);
    //printf("Consumer: testing...\n");    // add this to show wasted cycles
    while (global_availmsg<1) {	
      pthread_mutex_unlock(&mutex);
        
      // while buffer empty, wait...
      
      pthread_mutex_lock(&mutex);
      //printf("Consumer: testing...\n");    // add this to show wasted cycles      
    }
    
    // receive message
    int i = global_buffer[global_qout];
    global_qout += 1;
    if (global_qout>=QUEUE_SIZE) global_qout = 0; // wrap around
    global_availmsg -= 1;
    
    pthread_mutex_unlock(&mutex);    
      
    return(i);
}



// producer thread function
void *producer_thread(void *args) {
  
  // send a predefined number of messages
  for (int i=0;i<MESSAGES;i++) {
  
    printf("Producer: sending msg %d\n",i);  
    send_msg(i);
    
  }
      
  // exit and let be joined
  pthread_exit(NULL); 
}
  
  
// receiver thread function
void *consumer_thread(void *args) {
  
  // receive a predefined number of messages
  for (int i=0;i<MESSAGES;i++) {

    int msg = recv_msg();
    printf("Consumer: received msg %d\n",msg);
      
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
