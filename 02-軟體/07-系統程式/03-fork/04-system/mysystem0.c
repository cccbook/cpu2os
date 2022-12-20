#include <stdio.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/wait.h>

void mysystem(char *arg[]) {
  execvp(arg[0], arg); // child : exec("ls -l")
}

int main() {
  char *arg[] = {"ls", "-l", NULL };
  mysystem(arg);
  printf("main end!\n");
}
