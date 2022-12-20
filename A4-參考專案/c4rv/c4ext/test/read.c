#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char **argv) {
  int fd, len;
  char *buf, *oFile;
  oFile = argv[1];
  printf("oFile=%s\n", oFile);
  buf = malloc(10000);
  if ((fd = open(oFile, 0, 0)) < 0) { printf("could not open(%s)\n", oFile); return -1; }
  len = read(fd, buf, 10000);
  printf("len=%d\n", len);
  printf("buf=%s\n", buf);
  close(fd);
  return 0;
}
