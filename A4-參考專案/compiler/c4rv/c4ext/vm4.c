#include "c4.h"

int main(int argc, char **argv) { // 主程式
  oFile = argv[1];
  init();
  bc_load();
  vm_main(pc, argc, argv);
}
