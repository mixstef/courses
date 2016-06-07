#include <iostream>
#include <string>

using namespace std;

class Employee {
  string firstname;
  string lastname;
  int age;
  double salary;
public:
  Employee(const string &fs,const string &ls,int ag,double sal);
  string get_firstname() { return firstname; }
  string get_lastname() { return lastname; }
  int get_age() { return age; }
  double get_salary() { return salary; } 

  friend ostream &operator<<(ostream &stream,const Employee &empl);
};

Employee::Employee(const string &fs,const string &ls,int ag,double sal): firstname(fs),lastname(ls),age(ag),salary(sal) {

}

ostream &operator<<(ostream &stream,const Employee &empl) {
  // function must be declared friend of Employee for the following to work!
  stream << empl.firstname << " " << empl.lastname << "," << empl.age << "->" << empl.salary;
  return stream;	// this must be returned, in order to work with concatenated <<
}

int main() {
  Employee empl("John","Doe",34,12.98);
  cout << empl << endl;
  return 0;
}
