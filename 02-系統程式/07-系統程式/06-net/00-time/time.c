#include <stdio.h>
#include <unistd.h>
#include <time.h>

int main(int argc, char *argv[]) {
  while(1) {
    time_t ticks = time(NULL);
    printf("%.24s\n", ctime(&ticks));
    sleep(1);
  }
}
