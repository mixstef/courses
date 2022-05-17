// C++ simple serial quicksort implementation
// compile with: g++ -Wall -O2 -std=c++11 quicksort-simple.cpp -o quicksort-simple

#include <iostream>
#include <vector>
#include <random>	// for default_random_engine, uniform_real_distribution
#include <algorithm>	// for generate
#include <chrono>


constexpr size_t N = 10000000;

using namespace std;



void quicksort(vector<double>::iterator b,vector<double>::iterator e) {
 
  if ((e-b)<2) return;
  
  // get last element as pivot
  auto pivot_p = e-1;
    
  // partition elements
  auto i = b;
  for (auto j = b; j!=pivot_p; ++j) {
    if (*j<*pivot_p) swap(*i++,*j);
  }
  swap(*i,*pivot_p);
     
  // recursively sort halves
  quicksort(b,i);
  quicksort(i+1,e);
  
}


double rng() {
  thread_local static std::default_random_engine gen;		// this lives between invocations *per thread*
  std::uniform_real_distribution<double> dist(0.0,1.0);		// [0.0,1.0)
  
  return dist(gen);
}


int main() {

  // alloc vector
  vector<double> v(N);
  
  // initialize vectors
  std::generate(v.begin(),v.end(),rng);
  
  auto start = chrono::high_resolution_clock::now();

  // execute test load
  quicksort(v.begin(),v.end());
    
  auto stop = chrono::high_resolution_clock::now();
      
  // check results

  for (auto i=v.begin()+1;i!=v.end();++i) {
    if (*(i-1)>*i) {
      cout << "error " << endl;
      break;  
    } 
  }
    
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}
