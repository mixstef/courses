// Serial Matrix multiplication with TBB threading
// Compile with:  g++ -Wall -O2 -std=c++11 matmul-tbb2.cpp -o matmul-tbb2 -ltbb

#include <iostream>
#include <chrono>

#include "tbb/tbb.h"

size_t const N = 1000;	// all matrices are NxN 

using namespace std;


int main() {

  // alloc matrices
  double *a = new double[N*N];
  double *b = new double[N*N]; // assume B transposed
  double *c = new double[N*N];

  // init input (and output) matrices
  for (size_t i=0;i<N*N;i++) {
    a[i] = 2.0;
    b[i] = 3.0;
    c[i] = 20.0;
  }

  auto start = chrono::high_resolution_clock::now();

  // execute test load
  
  // for all rows of A and C
  tbb::parallel_for(tbb::blocked_range<size_t>(0,N),[&](tbb::blocked_range<size_t>&r) {
  
    for (size_t i=r.begin();i!=r.end();++i) {
      
      // for all columns of B
      for (size_t j=0;j<N;++j) { 
      
        // for row Ai and column Bj, outputs single element of Ci,j
        double sum = 0.0;
        for (size_t k=0;k<N;++k) { 
          sum += a[i*N+k]*b[j*N+k];	// NOTE: B is transposed
        }
        c[i*N+j] = sum;
    
      }
      
    }
  
  });

  auto stop = chrono::high_resolution_clock::now();
  
  // test results
  for (size_t i=0;i<N*N;i++) {
    if (c[i]!=6.0*N) { cout << "error!" << endl; break; }  
  }

  // free matrices
  delete[] a;
  delete[] b;
  delete[] c;
  
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}

