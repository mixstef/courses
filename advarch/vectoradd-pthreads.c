#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <pthread.h>

// Sample vector addding program using pthreads
// compile with:
// gcc -O2 -Wall -pthread vectoradd-pthreads.c -o vectoradd-pthreads -DTHREADS=16 -DN=100000 -DR=50000
// use R>=1000, N=100000, THREADS= 1,2,4,6,8,16,32

// size of vectors
//#define N 10000000

// how many threads we will create
//#define THREADS 32

// artificial repetitions per thread
//#define R 100

// how many elements a thread will process - NOTE: surrounding ()!
#define BLOCKSIZE  ((N+THREADS-1)/THREADS)

// struct of info passed to each thread
struct thread_params {
  double *a,*b;		// pointers to input vectors start
  double *c;		// pointer to output vector start
  int n;		// how many consecutive elements to process
};


// the thread work function
void *thread_func(void *args) {
struct thread_params *tparm;
double *pa,*pb,*pc;
int i,n;

  // thread input params
  tparm = (struct thread_params *)args;

  // for R artificial repetitions
  for (i=0;i<R;i++) {
    
    // process block of consecutive elements
    pa = tparm->a;
    pb = tparm->b;
    pc = tparm->c;
    n = tparm->n;
    while (n>0) {
      *pc = *pa+*pb;
      pa++; pb++; pc++;
      n--;
    }
  
  }
    
  // exit and let be joined
  pthread_exit(NULL); 
}


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}



int main() {
double *a,*b,*c;
int i,check;
double ts,te;
double checksum;

// array of structs to fill and pass to threads on creation
struct thread_params tparm[THREADS];
// table of thread IDs (handles) filled on creation, to be used later on join
pthread_t threads[THREADS];


  printf("Adding vectors with %d threads (%d doubles per thread)\n",THREADS,BLOCKSIZE);

  // allocate test arrays
  a = (double *)malloc(N*sizeof(double));
  if (a==NULL) exit(1);
  b = (double *)malloc(N*sizeof(double));
  if (b==NULL) { free(a); exit(1); }
  c = (double *)malloc(N*sizeof(double));
  if (c==NULL) { free(a); free(b); exit(1); }
  
  //initialize all arrays - cache warm-up
  for (i=0;i<N;i++) {
    a[i]=1.0; b[i]=2.0; c[i]=20.0;
  }
  
  // get starting time (double, seconds) 
  get_walltime(&ts);
 
  // create all threads
  check = 0;
  for (i=0;i<THREADS;i++) {
    // fill params for this thread
    tparm[i].a = a+i*BLOCKSIZE;
    tparm[i].b = b+i*BLOCKSIZE;
    tparm[i].c = c+i*BLOCKSIZE;
    if ((check+BLOCKSIZE)>=N) { // less than blocksize to do...
      tparm[i].n = N-check; 
    }
    else { 
      tparm[i].n = BLOCKSIZE; // there IS blocksize work to do! 
    }
    check += BLOCKSIZE;

    // create thread with default attrs (attrs=NULL)
    if (pthread_create(&threads[i],NULL,thread_func,&tparm[i])!=0) {
      printf("Error in thread creation!\n");
      exit(1);
    }   
  }
  
  // block on join of threads
  for (i=0;i<THREADS;i++) {
    pthread_join(threads[i],NULL);
  }  
 
  // get ending time
  get_walltime(&te);
    
  printf("Elapsed time: %f sec\n",(te-ts));
 
  // check results
  checksum = 0;
  for (i=0;i<N;i++) {
     checksum += c[i];
  }
  printf("Sum is %f (should be %f)\n",checksum,(N*3.0));
  
  // free arrays
  free(a); free(b); free(c);
  
  return 0;
}
