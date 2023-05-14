// Sample program to demonstrate basic operations on c++ vectors
// compile with: g++ -Wall -O2 -std=c++11 vector-test.cpp -o vector-test $(pkg-config --libs --cflags tbb)

#include <iostream>
#include <vector>
#include <random>	// for default_random_engine, uniform_real_distribution
#include <algorithm>	// for generate

#include "tbb/tbb.h"


constexpr size_t N = 10;

using namespace std;


double rng() {
  thread_local static std::default_random_engine gen;		// this lives between invocations *per thread*
  std::uniform_real_distribution<double> dist(0.0,1.0);		// [0.0,1.0)
  
  return dist(gen);
}


int main() {

  // alloc vector of size N
  vector<double> v(N);
  
  // initialize vector to random values (same for each program run)
  std::generate(v.begin(),v.end(),rng);

  // print values in vector via a range-base loop
  for (const double& x : v) { // x is a (const) reference to each element
    cout << x << ' ';
  }
  cout << endl;
  
  // get a single value in vector
  cout << v[1] << endl;
  cout << v.at(1) << endl;
  
  //cout << v[20] << endl; // this will print garbage
  //cout << v.at(20) << endl; // this throws an exception when out of range
  
  // set a single value of vector
  v[1] = 63.0;
  v.at(1) = 84.0;
  
  //v[111111111] = 63.0;	// will (probably) segfault
  //v.at(111111111) = 84.0;	// will throw an exception
  
  // add one element to the end of vector
  v.push_back(123.0);

  for (double x : v) {	// this makes a copy of elements to x, better use ref
    cout << x << ' ';
  }
  cout << endl;

  // remove last element
  v.pop_back();
  v.pop_back();
  
  for (auto&& x : v) {	// a "forwarding reference", type of x will be double&
    cout << x << ' ';
  }
  cout << endl;
  
  // an iterator to the first element of vector
  vector<double>::iterator itr1 = v.begin();
  cout << "v[0] is " << *itr1 << endl;
  
  // pointer arithmetics with iterators
  cout << "v[2] is " << *(itr1+2) << endl;
  
  
  // an iterator to the last element of vector
  auto itr2 = v.end()-1;
  cout << "v[last] is " << *itr2 << endl;
  
  //cout << *(itr2+111111111) << endl;	// will (probably) segfault
  
  // looping through vector by using iterators
  for (auto itr = v.begin(); itr != v.end(); ++itr) {
    cout << *itr << ' ';
  }
  cout << endl;  
  
  // vector looping with TBB parallel for
  tbb::parallel_for(tbb::blocked_range<vector<double>::iterator>(v.begin(),v.end()),
  [](const tbb::blocked_range<vector<double>::iterator>& r) {
    for (auto itr = r.begin(); itr != r.end(); ++itr) {
      *itr += 100;
    }
  });

  for (auto&& x : v) {
    cout << x << ' ';
  }
  cout << endl;
 
  return 0;
}
 
