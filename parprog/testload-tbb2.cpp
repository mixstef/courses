// C++ test load template for TBB usage
// compile with: g++ -Wall -O2 -std=c++11 testload-tbb2.cpp -o testload-tbb2 -ltbb

#include <iostream>
#include <chrono>

#include "tbb/tbb.h"


size_t const N = 10000000;
size_t const R = 10;


using namespace std;


// a sample map function on a[i]
double map_func(double x) {
  for (size_t j=0;j<R;++j) {
    x = x+j;
  }
  return x;
} 


int main() {

  // alloc array(s)
  double *a = new double[N];
  
  // init array(s)
  for (size_t i=0;i<N;++i) {
    a[i]=2.0*i;
  }
  
  auto start = chrono::high_resolution_clock::now();

  // execute test load
  tbb::parallel_for(tbb::blocked_range<size_t>(0,N),[&a](const tbb::blocked_range<size_t>& r) {
    for (size_t i=r.begin();i!=r.end();++i) {
      a[i] = map_func(a[i]);
    }
  });
    
  auto stop = chrono::high_resolution_clock::now();
      
  // check results

  for (size_t i=0;i<N;++i) {
    if (a[i]!=map_func(2.0*i)) {
      cout << "error " << endl;
      break;  
    }
  }
  
  // free array(s)
  delete[] a;
  
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}
