// 修改自 -- https://github.com/spencertipping/jit-tutorial/blob/master/jitproto.c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/mman.h>

typedef long(*fn)(long);

int main() {
  char *memory = mmap(NULL,             // address
                      4096,             // size
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS,
                      -1,               // fd (not used here)
                      0);               // offset (not used here)
  assert(memory != NULL);
  char code[] = {
    0x48, 0x8b, 0xc7 /*mov %rdi, %rax*/, 
    0xc3 /*ret*/
  };
  memcpy(memory, code, 4);
  fn f = (fn) memory;

  int i;
  for (i = 0; i < 10; ++i)
    printf("f(%d) = %ld\n", i, (*f)(i));
  munmap(f, 4096);
  return 0;
}