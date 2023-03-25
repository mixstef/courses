// Sample triad benchmark with arrays of doubles, using pthreads
// compile with : gcc -O2 -Wall -pthread triad-double-pthreads.c -o triad-double-pthreads -DTHREADS=4 -DN=10000 -DR=10000


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <pthread.h>


// how many elements a thread will process - note surrounding ()!
#define BLOCKSIZE  ((N+THREADS-1)/THREADS)

// struct of info passed to each thread
struct thread_params {
  double *b,*c,*d;		// pointers to input vectors start
  double *a;		// pointer to output vector start
  int n;		// how many consecutive elements to process
};


// the thread work function
void *thread_func(void *args) {

  // thread input params
  struct thread_params *tparm = (struct thread_params *)args;

  int n = tparm->n;
  double *a = tparm->a;
  double *b = tparm->b;
  double *c = tparm->c;
  double *d = tparm->d;

  // for R artificial repetitions
  for (int j=0;j<R;j++) {
    
    // process block of consecutive elements    
    for (int i=0;i<n;i++) {
      a[i] = b[i]*c[i]+d[i];
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
double ts,te,mflops;

// array of structs to fill and pass to threads on creation
struct thread_params tparm[THREADS];
// table of thread IDs (handles) filled on creation, to be used later on join
pthread_t threads[THREADS];


  printf("Computing vectors with %d threads (%d doubles per thread)\n",THREADS,BLOCKSIZE);

  // allocate test arrays
  double *a = (double *)malloc(N*sizeof(double));
  if (a==NULL) exit(1);
  double *b = (double *)malloc(N*sizeof(double));
  if (b==NULL) { free(a); exit(1); }
  double *c = (double *)malloc(N*sizeof(double));
  if (c==NULL) { free(a); free(b); exit(1); }
  double *d = (double *)malloc(N*sizeof(double));
  if (d==NULL) { free(a); free(b); free(c); exit(1); }
  
  //initialize all arrays - cache warm-up
  for (int i=0;i<N;i++) {
    a[i]=2.0*i;
    b[i]=-i;
    c[i]=i+5.0;
    d[i]=-7.0*i;
  }
  
  // get starting time (double, seconds) 
  get_walltime(&ts);
 
  // create all threads
  for (int i=0;i<THREADS;i++) {
    // fill params for this thread
    tparm[i].a = a+i*BLOCKSIZE;
    tparm[i].b = b+i*BLOCKSIZE;
    tparm[i].c = c+i*BLOCKSIZE;
    tparm[i].d = d+i*BLOCKSIZE;    
    
    if (i==(THREADS-1)) { // maybe less than blocksize to do...
      tparm[i].n = N-i*BLOCKSIZE; 
    }
    else { 
      tparm[i].n = BLOCKSIZE; // always blocksize work to do! 
    }

    // create thread with default attrs (attrs=NULL)
    if (pthread_create(&threads[i],NULL,thread_func,&tparm[i])!=0) {
      printf("Error in thread creation!\n");
      // free arrays
      free(a); free(b); free(c); free(d);
      exit(1);
    }   
  }
  
  // block on thread join
  for (int i=0;i<THREADS;i++) {
    if (pthread_join(threads[i],NULL)!=0) {
      printf("Error in thread join!\n");
      // free arrays
      free(a); free(b); free(c); free(d);
      exit(1);    
    }
  }  
 
  // get ending time
  get_walltime(&te);
    
  // check results
   for (int i=0;i<N;i++) {
    if (a[i]!=b[i]*c[i]+d[i]) {
      printf("Error!\n");
      break;
    }
  }
  
  // compute mflops/sec (2 floating point operations per R*N passes)
  mflops = (2.0*R*N)/((te-ts)*1e6);
  
  printf("MFLOPS/sec = %f\n",mflops);
    
  // free arrays
  free(a); free(b); free(c); free(d);
  
  return 0;
}
