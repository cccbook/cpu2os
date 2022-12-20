
## jitAdd

```
wsl> gcc jitAdd.c -o jitAdd
wsl> ./jitAdd
f(2,3) = 5
```

## 說明

```cpp
  char code[] = {
    0xf3,0x0f,0x1e,0xfa,      // endbr64
    0x8d,0x04,0x37,           // lea    (%rdi,%rsi,1),%eax
    0xc3                      // retq
  };
```

lea 是個 x86 上難以理解的指令，基本上就是 shift-and-add

leaq (%rdi, %rdi, 2), %rax 就是

rax = 2*%rdi+%rdi

所以上面的 lea (%rdi, %rsi, 1), %eax

就是 rax = 1*rsi+rdi

其中 rdi/rsi 分別放第 1/2 個參數

* https://stackoverflow.com/questions/46597055/using-lea-on-values-that-arent-addresses-pointers

lea (see Intel's instruction-set manual entry) is a shift-and-add instruction that uses memory-operand syntax and machine encoding. This explains the name, but it's not the only thing it's good for. It never actually accesses memory, so it's like using & in C.