// quicksort implementation via a thread worker pool
// compile with e.g.:
// gcc -O2 -Wall -pthread quicksort-threads-pool.c -o quicksort-threads-pool -DN=10000000 -DTHREADS=4

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}

// ---------- message queue -------------

struct message {
  double *a;	// a!=NULL -> work(a,n) msg
  int n;	// a==NULL: n==0 -> shutdown msg | n>0: work_termination(n) msg 
};


#define QUEUE_SIZE 1000

// global integer buffer
struct message global_buffer[QUEUE_SIZE];
int global_qin = 0;	// insertion index
int global_qout = 0;	// extraction index


// global avail messages count
int global_availmsg = 0;	// empty

// conditional variable, signals a put operation (receiver waits on this)
pthread_cond_t msg_in = PTHREAD_COND_INITIALIZER;
// conditional variable, signals a get operation (sender waits on this)
pthread_cond_t msg_out = PTHREAD_COND_INITIALIZER;

// mutex protecting common resources
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


void send(double *a,int n) {

    pthread_mutex_lock(&mutex);
    while (global_availmsg>=QUEUE_SIZE) { 
    
      pthread_cond_wait(&msg_out,&mutex);  
      
    }
    
    // send message
    global_buffer[global_qin].a = a;
    global_buffer[global_qin].n = n;
        
    global_qin += 1;
    if (global_qin>=QUEUE_SIZE) global_qin = 0; // wrap around
    global_availmsg += 1;
    
    // signal the receiver that something was put in buffer
    pthread_cond_signal(&msg_in);
    
    pthread_mutex_unlock(&mutex);

}


void recv(double **a,int *n) {

    // lock mutex
    pthread_mutex_lock(&mutex);
    while (global_availmsg<1) {	
    
      pthread_cond_wait(&msg_in,&mutex);  
    
    }
    
    // receive message
    *a = global_buffer[global_qout].a;
    *n = global_buffer[global_qout].n;    
    
    global_qout += 1;
    if (global_qout>=QUEUE_SIZE) global_qout = 0; // wrap around
    global_availmsg -= 1;
      
    // signal the sender that something was removed from buffer
    pthread_cond_signal(&msg_out);
    
    pthread_mutex_unlock(&mutex);

}


// --------- quicksort functions ----------------

#define CUTOFF 10


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


void quicksort(double *a,int n) {
int i;
  // check if below cutoff limit
  if (n<=CUTOFF) {
    inssort(a,n);
    return;
  }
  
  // partition into two halves
  i = partition(a,n);
   
  // recursively sort halves
  quicksort(a,i);
  quicksort(a+i,n-i);
  
}

// -------- thread worker function ---------

#define SPLIT_LIMIT 10000

void *work(void *args) {
double *a;
int i,n;

  
  // do until shutdown
  do {
    // receive msg from queue
    recv(&a,&n);	// blocking call

    if ((a==NULL)&&(n>0)) { // a termination logging msg, ignore (resend)
      send(a,n); // put back to queue
      continue;
    }
    if ((a==NULL)&&(n==0)) { // a shutdown msg
      send(a,n); // put back to queue
      break; // exit loop
    }

    if (n<=SPLIT_LIMIT) { // handle this ourselves
      quicksort(a,n);
      // send termination log
      send(NULL,n);
    }
    else {  // partition and let 2 new threads to sort sublists
      i = partition(a,n);
    
      // send 2 msgs to queue
      send(a,i);
      send(a+i,n-i);
     
    }
  } while (1);
  
  // exit and let be joined
  pthread_exit(NULL);   
}

// ---------- main program ---------

int main() {
double ts,te;
double *a;
int i;
 
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
  send(a,N);

  // track work completion messages
  int completed = 0;
  while (1) {
    double *a;
    int n;
    
    recv(&a,&n); // blocking call
    if ((a==NULL)&&(n>0)) { // a completion msg
      completed += n;
      if (completed==N) {
        // send shutdown msg and exit loop
        send(NULL,0);
        break;
      }
    }
    else { // push back in queue
      send(a,n);
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
    
  return 0;
}


