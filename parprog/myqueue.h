#include <pthread.h>

// queue instance control structure
typedef struct {
  size_t size;          // size of a single message
  size_t nmemb;         // size of queue (number of messages)
  void *buffer;         // ptr to queue circular storage
  size_t qin;           // insertion index
  size_t qout;          // extraction index
  size_t availmsg;      // current num of messages in queue
  pthread_mutex_t mutex;    // mutex protecting common resources
  pthread_cond_t msg_in;    // conditional variable, signals a put operation (receiver waits on this)
  pthread_cond_t msg_out;   // conditional variable, signals a get operation (sender waits on this)  
} myqueue_struct_t,*myqueue_t;


// queue function declarations
int myqueue_init(myqueue_t *queue,size_t size,size_t nmemb);
int myqueue_destroy(myqueue_t *queue);
int myqueue_send(myqueue_t *queue,void *msg);
int myqueue_recv(myqueue_t *queue,void *msg);

