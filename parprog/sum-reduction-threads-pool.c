// Sum reduction through recursive array splitting, using a thread worker pool.
// NOTE: Used as example only - NOT efficient way to perform sum reduction! 

// Compile with:
// gcc -O2 -Wall -pthread sum-reduction-threads-pool.c myqueue.c -o sum-reduction-threads-pool -DN=10000000 -DTHREADS=4

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#include "myqueue.h"


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


// ---------- message & done queue declarations -------------

#define WORK_QUEUE_SIZE 1000

// work diffusing message
struct work_msg {
  double *a;
  int n;	// n<=0 -> shutdown msg  
};

myqueue_t work_queue;


#define SIGNAL_QUEUE_SIZE 10

myqueue_t done_queue;

// work completion message
struct done_msg {
  double sum;   // partial sum
  int n;        // number of elements summed
};


// --------- reduction functions ----------------


#define SERIAL_THRESHOLD 10000


double serial_sum_reduction(double *a,int n) {

  double sum = 0.0;
  for (int i=0;i<n;i++) {
    sum += a[i];
  }
  
  return sum;  
}


void threaded_sum_reduction(double *a,int n) {

  // check if below serial threshold limit
  if (n<=SERIAL_THRESHOLD) { // handle this serially
    struct done_msg msg = {.sum = serial_sum_reduction(a,n), .n = n};
    // send work completion message to main 
    myqueue_send(&done_queue,&msg);
  }
  else {  
    
    int i = n/2;    // split in halves
    
    // create work diffusion message for left half
    struct work_msg msg = {.a = a, .n = i };
    myqueue_send(&work_queue,&msg);

    // handle right half ourselves
    threaded_sum_reduction(a+i,n-i);
  }
  
}

// -------- thread worker function ---------

void *work(void *args) {

  
  // do until shutdown
  do {
    // receive msg from work queue - blocking call
    struct work_msg msg;
    myqueue_recv(&work_queue,&msg);

    if (msg.n<=0) { // shutdown message (from main thread)
      break; 
    }
    
    // else, a work diffusing message was received
    threaded_sum_reduction(msg.a,msg.n);

  } while (1);
  
  // exit and let be joined
  pthread_exit(NULL);   
}

// ---------- main program ---------

int main() {
double ts,te;
double *a;
int i;

  myqueue_init(&work_queue,sizeof(struct work_msg),WORK_QUEUE_SIZE);
  myqueue_init(&done_queue,sizeof(struct done_msg),SIGNAL_QUEUE_SIZE);
  
 
  a = (double *)malloc(N*sizeof(double));
  if (a==NULL) {
    printf("error in malloc\n");
    exit(1);
  }

  // fill array with random numbers
  srand(0);
  for (i=0;i<N;i++) {
    a[i] = i+1;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  // table of thread IDs (handles) filled on creation, to be used later on join
  pthread_t threads[THREADS];

  // create pool threads
  for (int i=0;i<THREADS;i++) {

    // create thread with default attrs (attrs=NULL)
    if (pthread_create(&threads[i],NULL,work,NULL)) {
      printf("Error in thread creation!\n");
      exit(1);
    }   
  }
 
  // put first work message in queue
  struct work_msg msg = {.a = a, .n = N};
  myqueue_send(&work_queue,&msg);

  // track work completion messages
  int completed = 0;
  double sum = 0.0;
  while (1) {
    struct done_msg msg;    
    myqueue_recv(&done_queue,&msg); // blocking call
    sum += msg.sum;
    completed += msg.n;
    if (completed==N) {
      // send shutdown msgs and exit loop
      for (int i=0;i<THREADS;i++) {
        struct work_msg msg = {.a = NULL, .n = -i}; 
        myqueue_send(&work_queue,&msg);
      }
      break;
    }
  }  
  
  // block on thread join
  for (int i=0;i<THREADS;i++) {
    pthread_join(threads[i],NULL);
  }  

  // get ending time
  get_walltime(&te);

  // check result
  double result = ((double)N*(N+1))/2;  
  if (sum!=result) {
    printf("Reduction error!\n");
  }  

  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  myqueue_destroy(&work_queue);
  myqueue_destroy(&done_queue);
    
  return 0;
}


