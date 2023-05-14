// Serial fibonacci computation
// compile with: g++ -Wall -O2 -std=c++11 fibonacci.cpp -o fibonacci 

#include <iostream>
#include <chrono>



using namespace std;

constexpr long N = 35;

long fib(long n) {
  if (n<2) {
    return n;
  }
  
  long i,j;
  
  i = fib(n-1);
  j = fib(n-2);
  
  return i+j;
}


int main() {

 
  auto start = chrono::high_resolution_clock::now();

  // test load
  long f = fib(N);
      
  auto stop = chrono::high_resolution_clock::now();
  
  cout << f << endl;    
  
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}
