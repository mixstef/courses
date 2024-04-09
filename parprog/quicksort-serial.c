// Serial quicksort implementation
// compile with: gcc -O2 -Wall quicksort-serial.c -o quicksort-serial -DN=10000000

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#define CUTOFF 10


void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


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

  // sort array
  quicksort(a,N);
  
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

