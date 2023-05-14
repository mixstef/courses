// Base code for pipelining (no TBB used)
// compile with: g++ -Wall -O2 -std=c++11 pipeline-base.cpp -o pipeline-base

#include <iostream>
#include <chrono>
#include <thread>


using namespace std;


constexpr size_t N = 2;


int main() {

  size_t i = 0;  // simulate input state
  size_t j; // simulate output state
  bool fc_stop = false; // simulate flow control

  const auto stage1 = [&i](bool& fc_stop) -> int {
    this_thread::sleep_for(chrono::milliseconds(25));  // simulate load
    if (i<N) {
      return ++i;
    }
    else {
      fc_stop = true;
      return 0;
    }  
  };
  
  const auto stage2 = [](int x) -> int {
    this_thread::sleep_for(chrono::milliseconds(50));  // simulate load
    return 2*x; 
  };
  
  const auto stage3 = [](int x) -> int {
    this_thread::sleep_for(chrono::milliseconds(100));  // simulate load
    return x*x; 
  };

  const auto stage4 = [&j](int x) {
    this_thread::sleep_for(chrono::milliseconds(25));  // simulate load
    j = x;
  };


  auto start = chrono::high_resolution_clock::now();

  // execute test load
  while (true) {
    
    int t = stage1(fc_stop);
    if (fc_stop) break;	// no more input
    
    t = stage2(t);

    t = stage3(t);
    
    stage4(t);        
    
  }
    
  auto stop = chrono::high_resolution_clock::now();

 
  // check results
  if (j!=4*N*N) {
    cout << "error!" << endl; 
  }
        
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start);
  cout << 1e-6*duration.count() << " sec" << endl;

  return 0;
}
