// Testing for primes, serial version
// Compile with: g++ -Wall -O2 -std=c++11 pipeline-primes-base.cpp -o pipeline-primes-base

#include <iostream>
#include <cmath>
#include <chrono>


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

  
  auto start = chrono::high_resolution_clock::now();

  // stage1: i generator
  for (size_t i=2;i<=N;++i) {
     
    // stage2: find if i is prime
    pair<int,bool> p{i,isPrime(i)};
     
    // stage3: print i if prime
    if (p.second) {
      cout << p.first << endl;
    }
     
  }
    
  auto stop = chrono::high_resolution_clock::now();
  
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}
