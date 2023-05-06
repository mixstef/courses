// C++ triad benchmark without TBB threading
// compile with: g++ -Wall -O2 -std=c++11 triad.cpp -o triad

#include <iostream>
#include <chrono>



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
  for (size_t i=0;i<N;++i) {
    a[i] = b[i]*c[i]+d[i];
  }
    
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
