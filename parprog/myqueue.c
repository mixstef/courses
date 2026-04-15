// A sample thread-safe queue implementation

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

#include "myqueue.h"


int myqueue_init(myqueue_t *queue,size_t size,size_t nmemb) {
  if ((*queue = malloc(sizeof(myqueue_struct_t)))==NULL) {
    return ENOMEM;
  }
  
  (*queue)->size = size;
  (*queue)->nmemb = nmemb;
  
  if (((*queue)->buffer = malloc(size*nmemb))==NULL) {
    free(*queue);
    return ENOMEM;
  }
  
  (*queue)->qin = 0;
  (*queue)->qout = 0;
  (*queue)->availmsg = 0;
  
  pthread_mutex_init(&((*queue)->mutex),NULL);
  pthread_cond_init(&((*queue)->msg_in),NULL);
  pthread_cond_init(&((*queue)->msg_out),NULL);
    
  return 0;
}


int myqueue_destroy(myqueue_t *queue) {

  pthread_cond_destroy(&((*queue)->msg_out));
  pthread_cond_destroy(&((*queue)->msg_in));
  pthread_mutex_destroy(&((*queue)->mutex));
  
  free((*queue)->buffer);
  free(*queue);

  return 0;
}


int myqueue_send(myqueue_t *queue,void *msg) {

    pthread_mutex_lock(&((*queue)->mutex));
    while ((*queue)->availmsg >= (*queue)->nmemb) { 
    
      pthread_cond_wait(&((*queue)->msg_out),&((*queue)->mutex));  
      
    }
    
    // copy message in queue buffer
    memcpy((*queue)->buffer+(*queue)->qin*(*queue)->size,msg,(*queue)->size);
    
    (*queue)->qin += 1;
    if ((*queue)->qin >= (*queue)->nmemb) (*queue)->qin = 0; // wrap around
    (*queue)->availmsg += 1;
    
    // signal the receiver that something was put in buffer
    pthread_cond_signal(&((*queue)->msg_in));
    
    pthread_mutex_unlock(&((*queue)->mutex));
    
    return 0;
}


int myqueue_recv(myqueue_t *queue,void *msg) {

    pthread_mutex_lock(&((*queue)->mutex));
    while ((*queue)->availmsg < 1) {	
    
      pthread_cond_wait(&((*queue)->msg_in),&((*queue)->mutex));  
    
    }
    
    // copy message from buffer
    memcpy(msg,(*queue)->buffer+(*queue)->qout*(*queue)->size,(*queue)->size);
    
    (*queue)->qout += 1;
    if ((*queue)->qout >= (*queue)->nmemb) (*queue)->qout = 0; // wrap around
    (*queue)->availmsg -= 1;
      
    // signal the sender that something was removed from buffer
    pthread_cond_signal(&((*queue)->msg_out));
    
    pthread_mutex_unlock(&((*queue)->mutex));

    return 0;
}


