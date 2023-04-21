#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main() {
  srand(time(0));
  int i;
  for (i = 0; i < 10; i++) {
    printf("%d ", rand()%9);
  }
  return 0;
}
