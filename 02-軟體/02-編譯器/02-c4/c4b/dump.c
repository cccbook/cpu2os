#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <unistd.h>
#include <fcntl.h>
#define int long long

int pcode, codeLen, pdata, dataLen, pc, *code0;
char *data0;

void loadObj(char *objFile) {
  int f, iSize, codeSize;

  iSize = sizeof(int);
  codeSize = codeLen*iSize;

  f = open(objFile, 0);
  read(f, &pcode, iSize);
  read(f, &codeSize, iSize);
  read(f, &pdata, iSize);
  read(f, &dataLen, iSize);
  read(f, &pc, iSize);
  code0 = (int*) pcode;
  data0 = (char*) pdata;
  // read(f, pcode, codeSize);
  // read(f, data0, dataLen);
  close(f);
  printf("code0=%d codeSize=%d data0=%d dataLen=%d pc=%04x\n", code0, codeSize, data0, dataLen, pc);

}

int dasm() {
  int *p;
  p = 
}

int main() {
  loadObj("a.out");
}