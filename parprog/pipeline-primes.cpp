// Testing for primes with TBB pipeline
// Compile with: g++ -Wall -O2 -std=c++11 pipeline-primes.cpp -o pipeline-primes $(pkg-config --libs --cflags tbb)

#include <iostream>
#include <cmath>
#include <chrono>

#include "tbb/tbb.h"



using namespace std;


constexpr size_t N = 10000;



bool isPrime(int num) {

  int sq = sqrt(num);
  for (int i=2;i<=sq;++i) {
    if ((num % i)==0) return false;
  }
  return true;
}


int main() {

  size_t i = 1;
  
  const auto stage1 = [&i](tbb::flow_control& fc) -> int {
    if (i<N) {
      return ++i;
    }
    else {
      fc.stop();
      return 0;
    }  
  };  
  
  const auto stage2 = [](int num) -> pair<int,bool> {
    return pair<int,bool>{num,isPrime(num)};
  };
  
  const auto stage3 = [](pair<int,bool> p) {
    if (p.second) {
      cout << p.first << endl;
    }  
  };
  
  
  auto start = chrono::high_resolution_clock::now();

  tbb::parallel_pipeline(16,
    tbb::make_filter<void,int>(tbb::filter_mode::serial_in_order,stage1)&
    tbb::make_filter<int,pair<int,bool> >(tbb::filter_mode::parallel,stage2)&
    tbb::make_filter<pair<int,bool>,void>(tbb::filter_mode::serial_in_order,stage3)
  );
    
  auto stop = chrono::high_resolution_clock::now();
  
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}
