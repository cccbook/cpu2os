#include "vm.h"

int main(int argc, char *argv[]) {
  if (argc > 2 && strcmp(argv[2], "-d")==0) isDebug = 1;
  riscvInit();
  char *binFileName = argv[1];
  FILE *binFile = fopen(binFileName, "rb");
  mTop = fread(m, sizeof(uint8_t), MMAX, binFile);
  fclose(binFile);
  vm(m, mTop);
  return 0;
}
