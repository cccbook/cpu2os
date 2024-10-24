// 修改自 -- https://github.com/spencertipping/jit-tutorial/blob/master/jitproto.c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/mman.h>

typedef int(*fn)(int);

int main() {
  char *memory = mmap(NULL,             // address
                      4096,             // size
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS,
                      -1,               // fd (not used here)
                      0);               // offset (not used here)
  assert(memory != NULL);
  char code[] = {
   0xf3,0x0f,0x1e,0xfa,      // endbr64
   0x55               ,      // push   %rbp
   0x48,0x89,0xe5     ,      // mov    %rsp,%rbp
   0x53               ,      // push   %rbx
   0x48,0x83,0xec,0x18,      // sub    $0x18,%rsp
   0x89,0x7d,0xec     ,      // mov    %edi,-0x14(%rbp)
   0x83,0x7d,0xec,0x00,      // cmpl   $0x0,-0x14(%rbp)
   0x75,0x07          ,      // jne    1d <fib+0x1d>
   0xb8,0x00,0x00,0x00,0x00,      // mov    $0x0,%eax
   0xeb,0x2b          ,      // jmp    48 <fib+0x48>
   0x83,0x7d,0xec,0x01,      // cmpl   $0x1,-0x14(%rbp)
   0x75,0x07          ,      // jne    2a <fib+0x2a>
   0xb8,0x01,0x00,0x00,0x00,      // mov    $0x1,%eax
   0xeb,0x1e          ,      // jmp    48 <fib+0x48>
   0x8b,0x45,0xec     ,      // mov    -0x14(%rbp),%eax
   0x83,0xe8,0x01     ,      // sub    $0x1,%eax
   0x89,0xc7          ,      // mov    %eax,%edi
   0xe8,0xc9,0xff,0xff,0xff,      // callq  0 <fib>
   0x89,0xc3          ,      // mov    %eax,%ebx
   0x8b,0x45,0xec     ,      //  mov    -0x14(%rbp),%eax
   0x83,0xe8,0x02     ,      // sub    $0x2,%eax
   0x89,0xc7          ,      // mov    %eax,%edi
   0xe8,0xba,0xff,0xff,0xff,      // callq  0 <fib>
   0x01,0xd8          ,      // add    %ebx,%eax
   0x48,0x83,0xc4,0x18,      // add    $0x18,%rsp
   0x5b               ,      // pop    %rbx
   0x5d               ,      // pop    %rbp
   0xc3                      // retq
  };

  memcpy(memory, code, 0x50);
  fn f = (fn) memory;

  printf("f(5) = %d\n", (*f)(7));
  munmap(f, 4096);
  return 0;
}