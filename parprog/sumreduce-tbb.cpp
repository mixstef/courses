// Sum reduction - tbb version
// Compile with:  g++ -Wall -O2 -std=c++11 sumreduce-tbb.cpp -o sumreduce-tbb -ltbb

#include <iostream>
#include <chrono>

#include "tbb/tbb.h"

size_t const N = 100000000;

using namespace std;



int main() {

  // alloc array
  double *a = new double[N];
  
  // init array
  for (size_t i=0;i<N;++i) {
    a[i]=i+1;
  }
  
  auto start = chrono::high_resolution_clock::now();

  // execute test load
  double sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0,N),	// range
                                    0.0,	// identity
                                    [&a](const tbb::blocked_range<size_t>& r,double init) -> double {	// func
                                      double sum = init;
                                      for (size_t i=r.begin();i!=r.end();++i) {
                                        sum += a[i];    
                                      }
                                      return sum;
                                    },
                                    [](double x,double y) -> double {	// reduction
                                      return x+y;
                                    } );	// can use plus<double>() instead
  
  auto stop = chrono::high_resolution_clock::now();
      
  // check results
  if (sum!=((double)N*(N+1)/2)) {
      cout << "Reduction error: " << sum << endl;
  }

  // free array
  delete[] a;
  
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}
