// TBB parallel invoke template
// compile with: g++ -Wall -O2 -std=c++11 parallel-invoke-base.cpp -o parallel-invoke-base $(pkg-config --libs --cflags tbb)

#include <iostream>
#include <chrono>
#include <thread>

#include "tbb/tbb.h"



using namespace std;


void some_func(int x) {
   cout << x << endl;
   this_thread::sleep_for(chrono::milliseconds(25));  // simulate load
}


int main() {

 
  auto start = chrono::high_resolution_clock::now();

  // test load
  tbb::parallel_invoke(
    [](){ some_func(0); },
    [](){ some_func(1); },
    [](){ some_func(2); },
    [](){ some_func(3); }
  );
    
  auto stop = chrono::high_resolution_clock::now();
      
  
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;
  
  return 0;
}
