// Simple examples of lambda expressions
// compile with: g++ -Wall -O2 -std=c++11 lambda-example.cpp -o lambda-example

#include <iostream>


using namespace std;


int main() {

  // define a lambda
  auto lambda = [](int x) {
    return x+33;
  };
  
  // use lambda
  cout << lambda(5) << endl;
  
  //-----------------------------
  
  int k = 77;
  
  // define a lambda, k is captured by value
  auto lambda2 = [k](int x) {
    return x+33+k;
  };
  
  cout << lambda2(5) << endl;

  //-----------------------------

  // define a lambda, k is captured by value, allowed to change inside lambda capture
  auto lambda3 = [k](int x) mutable {
    k +=3;
    return x+33+k;
  };
  
  cout << lambda3(5) << endl;
  cout << lambda3(5) << endl;
  cout << k << endl;

  //-----------------------------

  // define a lambda, k is captured by reference
  auto lambda4 = [&k](int x) {
    k +=3;
    return x+33+k;
  };
  
  cout << lambda4(5) << endl;
  cout << lambda4(5) << endl;
  cout << k << endl;


  //-----------------------------

  // define a lambda, everything is captured by reference
  auto lambda5 = [&](int x) {
    k +=3;
    return x+33+k;
  };
  
  cout << lambda5(5) << endl;
  cout << lambda5(5) << endl;
  cout << k << endl;

  //-----------------------------

  // define a lambda, everything is captured by value
  auto lambda6 = [=](int x) {
    return x+33+k;
  };
  
  cout << lambda6(5) << endl;
  cout << lambda6(5) << endl;
 
  return 0;
}
