// C++ program to find min and max element of a vector of doubles.
// compile with: g++ -Wall -O2 -std=c++17 minmax.cpp -o minmax

#include <iostream>
#include <vector>
#include <random>	// for default_random_engine, uniform_real_distribution
#include <algorithm>	// for generate
#include <chrono>


constexpr size_t N = 100000000;

using namespace std;


double rng() {
  thread_local static std::default_random_engine gen;		// this lives between invocations *per thread*
  std::uniform_real_distribution<double> dist(0.0,1.0);		// [0.0,1.0)
  
  return dist(gen);
}


int main() {

  // alloc vector
  vector<double> v(N);
  
  // initialize vector
  std::generate(v.begin(),v.end(),rng);
  
  
  auto start = chrono::high_resolution_clock::now();

  // execute test load
  double minval = v.at(0);
  double maxval = v.at(0);
  for (auto p=v.begin()+1;p!=v.end();++p) {
    if (*p<minval) minval = *p;
    if (*p>maxval) maxval = *p;    
  }
    
  auto stop = chrono::high_resolution_clock::now();
  
      
  // check results

  const auto [minp, maxp] = std::minmax_element(v.begin(),v.end());
  
  if ((*minp!=minval)||(*maxp!=maxval)) {
    cout << "error " << endl;
  }
    
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}

