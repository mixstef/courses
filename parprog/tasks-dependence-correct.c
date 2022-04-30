// following code needs OpenMP4.0, i.e. running a sample OpenMP program
// with OMP_DISPLAY_ENV="TRUE" should display _OPENMP = '201307' or newer

// compile with: gcc -fopenmp -O2 -Wall tasks-dependence-correct.c -o tasks-dependence-correct

#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#include <omp.h>


int main() {
int x = 0;
int y = 0;
  #pragma omp parallel num_threads(3)
  {
  
    #pragma omp single
    {
      
      #pragma omp task depend(out:x)
      {
        printf("Thread %d: x=%d y=%d\n",omp_get_thread_num(),x,y);
        x = 33;
      }

      #pragma omp task depend(in:x) depend(out:y)
      {
        printf("Thread %d: x=%d y=%d\n",omp_get_thread_num(),x,y);
        y = x+66;
      }
      
      #pragma omp task depend(in:x,y)
      {
        printf("Thread %d: x=%d y=%d\n",omp_get_thread_num(),x,y);
        int t = y+1;
        printf("Thread %d: t=%d\n",omp_get_thread_num(),t);
      }
      
    }
    
  }

  return 0;
}
