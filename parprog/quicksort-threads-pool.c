// quicksort implementation via a thread worker pool
// compile with:
// gcc -O2 -Wall -pthread quicksort-threads-pool.c myqueue.c -o quicksort-threads-pool -DN=10000000 -DTHREADS=4

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


// ---------- message/signaling queue declarations -------------

#define WORK_QUEUE_SIZE 1000

// work diffusing message
struct message {
  double *a;
  int n;	// n<=0 -> shutdown msg  
};

myqueue_t work_queue;


#define SIGNAL_QUEUE_SIZE 10

myqueue_t signal_queue;


// --------- quicksort functions ----------------

#define CUTOFF 10

#define SERIAL_THRESHOLD 10000


void inssort(double *a,int n) {
int i,j;
double t;
  
  for (i=1;i<n;i++) {
    j = i;
    while ((j>0) && (a[j-1]>a[j])) {
      t = a[j-1];  a[j-1] = a[j];  a[j] = t;
      j--;
    }
  }

}


int partition(double *a,int n) {
int first,last,middle;
double t,p;
int i,j;

  // take first, last and middle positions
  first = 0;
  middle = n/2;
  last = n-1;  
  
  // put median-of-3 in the middle
  if (a[middle]<a[first]) { t = a[middle]; a[middle] = a[first]; a[first] = t; }
  if (a[last]<a[middle]) { t = a[last]; a[last] = a[middle]; a[middle] = t; }
  if (a[middle]<a[first]) { t = a[middle]; a[middle] = a[first]; a[first] = t; }
    
  // partition (first and last are already in correct half)
  p = a[middle]; // pivot
  for (i=1,j=n-2;;i++,j--) {
    while (a[i]<p) i++;
    while (p<a[j]) j--;
    if (i>=j) break;

    t = a[i]; a[i] = a[j]; a[j] = t;      
  }
  
  // return position of pivot
  return i;
}


void serial_quicksort(double *a,int n) {
int i;
  // check if below cutoff limit
  if (n<=CUTOFF) {
    inssort(a,n);
    return;
  }
  
  // partition into two halves
  i = partition(a,n);
   
  // recursively sort halves
  serial_quicksort(a,i);
  serial_quicksort(a+i,n-i);
  
}


void threaded_quicksort(double *a,int n) {

  // check if below serial threshold limit
  if (n<=SERIAL_THRESHOLD) { // handle this serially
    serial_quicksort(a,n);
    // send work termination message to main 
    myqueue_send(&signal_queue,&n);
  }
  else {  
    // partition into two halves
    int i = partition(a,n);
   
    // create work diffusion message for left half
    struct message msg;
    msg.a = a;
    msg.n = i;
    myqueue_send(&work_queue,&msg);
    
    // handle righthalf ourselves
    threaded_quicksort(a+i,n-i);
  }
  
}

// -------- thread worker function ---------

void *work(void *args) {

  
  // do until shutdown
  do {
    // receive msg from queue - blocking call
    struct message msg;
    myqueue_recv(&work_queue,&msg);

    if (msg.n<=0) { // shutdown message (from main thread)
      break; 
    }
    
    // else, a work diffusing message was received
    threaded_quicksort(msg.a,msg.n);

  } while (1);
  
  // exit and let be joined
  pthread_exit(NULL);   
}

// ---------- main program ---------

int main() {
double ts,te;
double *a;
int i;

  myqueue_init(&work_queue,sizeof(struct message),WORK_QUEUE_SIZE);
  myqueue_init(&signal_queue,sizeof(int),SIGNAL_QUEUE_SIZE);
  
 
  a = (double *)malloc(N*sizeof(double));
  if (a==NULL) {
    printf("error in malloc\n");
    exit(1);
  }

  // fill array with random numbers
  srand(0);
  for (i=0;i<N;i++) {
    a[i] = (double)rand()/RAND_MAX;
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
  struct message msg;
  msg.a = a;
  msg.n = N;
  myqueue_send(&work_queue,&msg);

  // track work completion messages
  int completed = 0;
  while (1) {
    int n;    
    myqueue_recv(&signal_queue,&n); // blocking call
    completed += n;
    if (completed==N) {
      // send shutdown msgs and exit loop
      for (int i=0;i<THREADS;i++) {
        struct message msg;
        msg.n = -i;
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

  // check sorting
  for (i=0;i<(N-1);i++) {
    if (a[i]>a[i+1]) {
      printf("Sort failed!\n");
      break;
    }
  }  

  free(a);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  myqueue_destroy(&work_queue);
  myqueue_destroy(&signal_queue);
    
  return 0;
}


