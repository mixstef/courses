// blank benchmark template

// compile with: gcc -Wall -O2 template.c -o template
// check assembly output: gcc -Wall -O2 template.c -S



#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}




int main() {
  double ts,te;
  
  // 1. δέσμευση πινάκων
  
  // 2. αρχικοποίηση πινάκων


  // get starting time (double, seconds) 
  get_walltime(&ts);
  
  // 3. δοκιμαστικό φορτίο

  // get ending time
  get_walltime(&te);
  
  // 4. έλεγχος πράξεων
  
  // 5. αποδέσμευση πινάκων

  printf("Exec Time (sec) = %f\n",te-ts);
  
  return 0;
}
