// Computes the "histogram" for 4 ranges of N float values between [0..1).
// The 4 ranges are [0..C0),[C0..C1),[C1..C2),[C2..1) where C0, C1 and C2
// are arbitrary float constants.
// Histogram computation is repeated R times
// compile with: gcc -Wall -O2 quad-hist-float.c -o quad-hist-float

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// προσθέστε με #include το κατάλληλο header file για εντολές SSE/AVX


#define N 10000
#define R 10000

#define C0 0.25
#define C1 0.5
#define C2 0.75


void get_walltime(double *wct) {

  // Μπορείτε να αλλάξετε το περιεχόμενο της get_walltime (π.χ. για λόγους
  // συμβατότητας με τα Windows)

  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


float *allocate_array() {

  // Αλλάξτε το περιεχόμενο της allocate_array για να εξασφαλίσετε
  // ευθυγράμμιση διευθύνσεων στα 32 bytes

  return (float *)malloc(N*sizeof(float));

}


void free_array(float *a) {

  // Μπορείτε να αλλάξετε το περιεχόμενο της free_array (π.χ. για λόγους
  // συμβατότητας με τα Windows)
  
  free(a);

}


void compute_histogram(float *a, float *results) {

  for (int j=0;j<R;j++) {  
  
    // Αντικαταστήστε το περιεχόμενο του loop του j έτσι ώστε να
    // υπολογίζετε το ιστόγραμμα με εντολές AVX/SSE
    
    results[0] = results[1] = results[2] = results[3] = 0;
    for (int i=0;i<N;i++) {
      if (a[i]<C0) results[0]++;
      else  if (a[i]<C1) results[1]++;
      else  if (a[i]<C2) results[2]++;
      else results[3]++;
    }    
  } 
  
}



int main() {
  double ts,te;
  
  // allocate input array
  float *a = allocate_array();
  if (a==NULL) {
    printf("Allocation failed!\n");
    exit(1);
  }

  // init array and prepare validation results
  float check[4] = {0.0};
  for (int i=0;i<N;i++) {
    a[i] = (rand()%100)/(float)100;
    if (a[i]<C0) check[0]++;
    else if (a[i]<C1) check[1]++;
    else if (a[i]<C2) check[2]++;
    else check[3]++;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  // workload
  float results[4];
  compute_histogram(a,results);

  // get ending time
  get_walltime(&te);

  // check results
  if ((results[0]!=check[0])||(results[1]!=check[1])||(results[2]!=check[2])||(results[3]!=check[3])) printf("Error!\n");
 
  // free input array
  free_array(a);

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
