// compile with: gcc -fopenmp -O2 -Wall tasks-dependence-initial.c -o tasks-dependence-initial

#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#include <omp.h>


int main() {
int x = 0;
int y = 0;

  #pragma omp parallel num_threads(3)
  {
  
    #pragma omp single nowait
    {
        // this will be task A
        printf("(A) Thread %d: x=%d y=%d\n",omp_get_thread_num(),x,y);
        x = 33;

        // this will be task B
        printf("(B) Thread %d: x=%d y=%d\n",omp_get_thread_num(),x,y);
        y = x+66;

        // this will be task C 
        printf("(C) Thread %d: x=%d y=%d\n",omp_get_thread_num(),x,y);
        int t = y+1;
        printf("(C) Thread %d: t=%d\n",omp_get_thread_num(),t);
      
    }
    
  }

  return 0;
}
