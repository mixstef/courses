// C++ triad benchmark with TBB threading
// compile with: g++ -Wall -O2 -std=c++11 triad-tbb1.cpp -o triad-tbb1 -ltbb

#include <iostream>
#include <chrono>

#include "tbb/tbb.h"


size_t const N = 100000000;


using namespace std;



int main() {

  // alloc array(s)
  double *a = new double[N];
  double *b = new double[N];
  double *c = new double[N];
  double *d = new double[N];
        
  // init array(s)
  for (size_t i=0;i<N;++i) {
    a[i]=2.0*i;
  }
  
  auto start = chrono::high_resolution_clock::now();

  // execute test load
  tbb::parallel_for(size_t(0),N,[&](size_t& i) {
    a[i] = b[i]*c[i]+d[i];
  });
    
  auto stop = chrono::high_resolution_clock::now();
      
  // check results

  for (size_t i=0;i<N;++i) {
    if (a[i]!=b[i]*c[i]+d[i]) {
      cout << "error " << endl;
      break;  
    }
  }
  
  // free array(s)
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}
