// Sum reduction - serial version
// Compile with:  g++ -Wall -O2 -std=c++11 sumreduce-serial.cpp -o sumreduce-serial

#include <iostream>
#include <chrono>


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
  double sum = 0.0;
  for (size_t i=0;i<N;++i) {
    sum += a[i];
  }
  
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
