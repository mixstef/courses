// pi integral computation - tbb version
// Compile with:  g++ -Wall -O2 -std=c++11 pi-integral-tbb.cpp -o pi-integral-tbb -ltbb

#include <iostream>
#include <chrono>

#include "tbb/tbb.h"

size_t const N = 100000000;	// integration steps 

using namespace std;



int main() {

  // computation of pi
  double w = 1.0/N;
  
  auto start = chrono::high_resolution_clock::now();

  double pi =  tbb::parallel_reduce(tbb::blocked_range<size_t>(1,N+1),
                                    0.0,
                                    [=](const tbb::blocked_range<size_t>& r,double init) -> double {
                                      double sum = init;
                                      for (size_t i=r.begin();i!=r.end();++i) {
                                        double x = w*(i-0.5);
                                        sum += w*4.0/(1.0+x*x);    
                                      }
                                      return sum;
                                    },
                                    [](double x,double y) -> double {
                                      return x+y;
                                    });

  auto stop = chrono::high_resolution_clock::now();
    
  cout.precision(10); // NOTE: for default format (not fixed), counts all digits
  cout << "pi = " << fixed << pi << endl;

  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
 
  return 0;
}



