// Example calculating pi value using integration (threaded version).
// Compile with: gcc -O2 -Wall -pthread pi-integral-threads.c -o pi-integral-threads -DN=10000000 -DTHREADS=4


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#include <pthread.h>


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


// how many elements a thread will process - NOTE: surrounding ()!
#define BLOCKSIZE  ((N+THREADS-1)/THREADS)


// struct of info passed to each thread
struct thread_params {
  double sum;	// output for each thread
  int start;	// beginning elements
  int n;	// number of elements to process
  double w;	// integration step
};



// sample thread function
void *thread_func(void *args) {

  struct thread_params *tp = (struct thread_params*)args;
  int start = tp->start;
  int end = tp->start+tp->n;
  double w = tp->w;
  
  // compute local sum of slices assigned to us

  double mysum = 0.0;	// private to each thread
  for (int i=start;i<end;i++) {
    double x = w*(i-0.5);	// midpoint
    mysum += 4.0/(1.0+x*x); // NOTE: without mult by step (w), done later
  }
  
  // write partial sum
   
  tp->sum = w*mysum;  
  
  // exit and let be joined
  pthread_exit(NULL); 
}


int main() {
double ts,te;

  pthread_t pid[THREADS];
  
  struct thread_params tparm[THREADS];


  double w = 1.0/N;	// integration step


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // for all threads
  for (int i=0;i<THREADS;i++) {
    // fill i-th member of tparm array
    tparm[i].start = i*BLOCKSIZE+1;	//  NOTE: +1 to get correct results!
    tparm[i].w = w;

    if (i==(THREADS-1)) { // last thread, maybe less than blocksize to do...
      tparm[i].n = N-i*BLOCKSIZE; 
    }
    else { 
      tparm[i].n = BLOCKSIZE; // always blocksize work to do! 
    }
    
    // create i-th thread, pass ptr to tparm[i]
    if (pthread_create(&pid[i],NULL,thread_func,&tparm[i])!=0) {
      printf("Error in thread creation!\n");
      exit(1);
    }
  }


  // block until join
  for (int i=0;i<THREADS;i++) {
    if (pthread_join(pid[i],NULL)!=0) {
      printf("Error in thread join!\n");
      exit(1);  
    }
  }

  // main thread computes final pi value
  double pi = 0.0;
  for (int i=0;i<THREADS;i++) {
    pi += tparm[i].sum;
  }

  // get ending time
  get_walltime(&te);

  
  printf("Computed pi=%.10f\n",pi);
  printf("Exec Time (sec) = %f\n",te-ts);

  return 0;
}
