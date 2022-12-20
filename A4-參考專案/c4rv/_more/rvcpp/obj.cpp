#include "obj.hpp"

int main(int argc, char *argv[]) {
  char *oFile = argv[1];
  obj_load();
  obj_dump();
}
