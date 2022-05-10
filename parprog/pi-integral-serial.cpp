// pi integral computation - serial version
// Compile with:  g++ -Wall -O2 pi-integral-serial.cpp -o pi-integral-serial

#include <iostream>
#include <chrono>

size_t const N = 100000000;	// integration steps 

using namespace std;



int main() {

  // computation of pi
  double w = 1.0/N;
  double pi = 0.0;
  double x;

  auto start = chrono::high_resolution_clock::now();

  for (size_t i=1;i<=N;++i) {
    x = w*(i-0.5);
    pi += w*4.0/(1.0+x*x);
  }

  auto stop = chrono::high_resolution_clock::now();
    
  cout.precision(10); // NOTE: for default format (not fixed), counts all digits
  cout << "pi = " << fixed << pi << endl;

  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
 
  return 0;
}



