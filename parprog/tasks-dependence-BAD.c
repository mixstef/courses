// compile with: gcc -fopenmp -O2 -Wall tasks-dependence-BAD.c -o tasks-dependence-BAD

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
      
      #pragma omp task
      {
        printf("Thread %d: x=%d y=%d\n",omp_get_thread_num(),x,y);
        x = 33;
      }

      #pragma omp task
      {
        printf("Thread %d: x=%d y=%d\n",omp_get_thread_num(),x,y);
        y = x+66;
      }
      
      #pragma omp task
      {
        printf("Thread %d: x=%d y=%d\n",omp_get_thread_num(),x,y);
        int t = y+1;
        printf("Thread %d: t=%d\n",omp_get_thread_num(),t);
      }
      
    }
    
  }

  return 0;
}
