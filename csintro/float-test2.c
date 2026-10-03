#include <stdio.h>
#include <stdlib.h>

int main() {
  float x;
  unsigned int *p = (unsigned int *)&x;

  *p = 0x3DCCCCCD;
  printf("%1.30f\n",x);

  return 0;
}