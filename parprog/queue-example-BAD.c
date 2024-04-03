// ERRONEOUS example of a queue implementation (no sync).
// Compile with: gcc -O2 -Wall -pthread queue-example-BAD.c -o queue-example-BAD

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


// ---- send/receive functions ----

void send_msg(int msg) {

    while (global_availmsg>=QUEUE_SIZE) { 
    
      // while buffer full, wait...
      
    }
    
    // send message
    global_buffer[global_qin] = msg;
    global_qin += 1;
    if (global_qin>=QUEUE_SIZE) global_qin = 0; // wrap around
    global_availmsg += 1;
    
}


int recv_msg() {

    while (global_availmsg<1) {	
    
      // while buffer empty, wait...
      
    }
    
    // receive message
    int i = global_buffer[global_qout];
    global_qout += 1;
    if (global_qout>=QUEUE_SIZE) global_qout = 0; // wrap around
    global_availmsg -= 1;
      
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


  return 0;
}
