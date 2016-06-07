// example program for the 'guess the word' game

#include <iostream>
#include <string>

using namespace std;


int checkchar(const string &,string &,char);
void printstate(const string &,const string &,const string &,const string &,int);


int main() {
  int tries = 4;	// tries left
  string word =  "reverberation";	// hidden word
  
  int wordl = word.length();
  string state = string(wordl,'0'); // hidden-shown positions, init to "000...0"
  int shown = 0;	// current number of uncovered positions
  string passed;	// every char entered so far (correct or not)
  string wrongs;	// wrong chars entered so far
  string rights;	// correct chars entred so far
  
  shown = checkchar(word,state,word[0]);
  shown+= checkchar(word,state,word[wordl-1]);

  rights += word[0];
  rights += word[wordl-1];
  passed = rights;
  
  //display initial state
  printstate(word,state,rights,wrongs,tries);
  
  char ch;  
  bool stillplaying = true;

  while (stillplaying) {
        
    // get use choice
    cout << "enter your choice:";
    cin >> ch;
    // check if already passed, if so, ask again
    while (passed.find(ch)!=string::npos) {
      cout << "passed already!\nenter your choice:";
      cin >> ch;
    }
    
    // correct or not, add char to passed
    passed += ch;

    // check if char in word
    int n = checkchar(word,state,ch);

    if (n>0) {	// found!
      rights += ch;
      shown += n;
    }
    else { // not found!
      wrongs += ch;
      --tries;
    }
    
    // display current state
    printstate(word,state,rights,wrongs,tries);
    
    // check for finishing conditions
    if (shown==wordl) {
      cout << "You won!" << endl;
      stillplaying = false;
    }
    
    if (tries<=0) {
      cout << "You have lost...sorry!" << endl;
      stillplaying = false;    
    }
  
  }
  
  return 0;
}

// function to check if a char exists in a word.
// Updates shown letters string (by putting 1 in uncovered positions).
// Returns number of places uncovered or 0
int checkchar(const string &word,string &state,char ch) {
  
  int found = 0;

  for (unsigned int i=0;i<word.length();++i) {	
    if (word[i]==ch) {	// found! 
      ++found;
      state[i] = '1';
    }
  }
  
  return found;
}

// function to print game state
void printstate(const string &word,const string &state,const string &rights,const string &wrongs,int tries) {
  
  for (unsigned int i=0;i<word.length();++i) {
    if (state[i]=='1') cout << word[i] << ' ';
    else cout << "_ ";
  }
  cout << endl;
  cout << "Correct: " << rights << "\tWrong: " << wrongs << "\tTries left: " << tries << endl; 

}
