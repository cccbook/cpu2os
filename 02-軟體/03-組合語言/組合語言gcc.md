# gcc 組合語言

## x86 組合語言

* x86 組合語言 -- http://www.int80h.org/

## MinGW 程式

檔案: add.c 

```c
int add(int a, int b) {
  int t = a, x=b;
  return a+b;
}
```

檔案: main.c

```c
#include <stdio.h>

extern int add(int a, int b);

int main() {
  int t = add(5, 8);
  printf("add(5, 8)=%d\n", t);
  return 0;
}
```

執行

```
$ gcc main.c add.c -o add        
$ ./add
add(5, 8)=13
$ gcc -fverbose-asm -S main.c -o 
main.s
$ gcc -fverbose-asm -S add.c -o add.s
$ gcc main.s add.s -o add        
$ ./add
add(5, 8)=13
```

組合語言 add.s

```asm
	.file	"add.c"
	.text
	.globl	_add
	.def	_add;	.scl	2;	.type	32;	.endef
_add:
	pushl	%ebp	 #
	movl	%esp, %ebp	 #,
	subl	$16, %esp	 #,
	movl	8(%ebp), %eax	 # a, tmp89
	movl	%eax, -4(%ebp)	 # tmp89, t
	movl	12(%ebp), %eax	 # b, tmp90
	movl	%eax, -8(%ebp)	 # tmp90, x
	movl	8(%ebp), %edx	 # a, tmp91
	movl	12(%ebp), %eax	 # b, tmp92
	addl	%edx, %eax	 # tmp91, D.1490
	leave
	ret
	.ident	"GCC: (tdm-1) 5.1.0"

```

組合語言 main.s

```asm
	.def	___main;	.scl	2;	.type	32;	.endef
	.section .rdata,"dr"
LC0:
	.ascii "add(5, 8)=%d\12\0"
	.text
	.globl	_main
	.def	_main;	.scl	2;	.type	32;	.endef
_main:
	pushl	%ebp	 #
	movl	%esp, %ebp	 #,
	andl	$-16, %esp	 #,
	subl	$32, %esp	 #,
	call	___main	 #
	movl	$8, 4(%esp)	 #,
	movl	$5, (%esp)	 #,
	call	_add	 #
	movl	%eax, 28(%esp)	 # tmp89, t
	movl	28(%esp), %eax	 # t, tmp90
	movl	%eax, 4(%esp)	 # tmp90,
	movl	$LC0, (%esp)	 #,
	call	_printf	 #
	movl	$0, %eax	 #, D.1933
	leave
	ret
	.ident	"GCC: (tdm-1) 5.1.0"
	.def	_add;	.scl	2;	.type	32;	.endef
	.def	_printf;	.scl	2;	.type	32;	.endef

```


## 解說

add.s

```
  .file  "add.c"            // 原始碼是 add.c
  .text                     // 程式段開始
  .globl  _add              // _add 是個全域 Global 標記
  .def  _add;  .scl  2;  .type  32;  .endef
  // 32 位元，參考 https://stackoverflow.com/questions/1317081/gccs-assembly-output-of-an-empty-program-on-x86-win32
_add:                       // _add 標記
  pushl  %ebp   #           // 將 ebp 框架暫存器推入堆疊
  movl  %esp, %ebp   #,     // ebp = esp, 新的框架位址 
  movl  8(%ebp), %edx   # a, tmp89   // 將 a 放入 edx
  movl  12(%ebp), %eax   # b, tmp90  // 將 b 放入 eax
  addl  %edx, %eax   # tmp89, D.1488 // eax = a+b
  popl  %ebp   #            // 恢復呼叫前的 ebp
  ret                       // 返回上一層
  .ident  "GCC: (tdm-1) 5.1.0"
```

* https://eli.thegreenplace.net/2011/02/04/where-the-top-of-the-stack-is-on-x86

繪圖說明 call/ret 的原理

* https://segmentfault.com/q/1010000007057142

```
leave
ret
```

相當於

```
movl    %ebp, %esp
popl    %ebp
popl    %eip
```

* https://stackoverflow.com/questions/20129107/what-is-the-x86-ret-instruction-equivalent-to


## 主程式解說

```
  .file  "main.c"
  .def  ___main;  .scl  2;  .type  32;  .endef
  .section .rdata,"dr"
LC0:
  .ascii "add(5, 8)=%d\12\0"
  .text
  .globl  _main
  .def  _main;  .scl  2;  .type  32;  .endef
_main:
  pushl  %ebp   #
  movl  %esp, %ebp   #,
  andl  $-16, %esp   #,
  subl  $32, %esp   #,              // 框架大小分配 32
  call  ___main   #
  movl  $8, 4(%esp)   #,            // 準備參數 b=8
  movl  $5, (%esp)   #,             //         a=5
  call  _add   #                    // 呼叫 add(a,b)
  movl  %eax, 28(%esp)   # tmp89, t // t = eax
  movl  28(%esp), %eax   # t, tmp90
  movl  %eax, 4(%esp)   # tmp90,    // 將 t 放到堆疊
  movl  $LC0, (%esp)   #,           // 將 LC0 放入堆疊
  call  _printf   #                 // 呼叫 _printf
  movl  $0, %eax   #, D.1933        // 設定傳回值 eax 為 0
  leave                             // 準備離開，恢復框架
  ret
  .ident  "GCC: (tdm-1) 5.1.0"
  .def  _add;  .scl  2;  .type  32;  .endef
  .def  _printf;  .scl  2;  .type  32;  .endef
```

說明: https://stackoverflow.com/questions/7060970/substitutes-for-x86-assembly-call-instruction



## 遞迴範例

* https://github.com/ccccourse/sp/tree/master/code/c/03-asmVm/gcc/03-fib

C 語言: fib.c

```c
int fib(int n) {
  if (n <= 1) return 1;
  return fib(n-1) + fib(n-2);
}

```

組合語言:  fib.s

```
	.file	"fib.c"
	.text
	.globl	_fib
	.def	_fib;	.scl	2;	.type	32;	.endef
_fib:
	pushl	%ebp	 #
	movl	%esp, %ebp	 #,
	pushl	%ebx	 #
	subl	$20, %esp	 #,
	cmpl	$1, 8(%ebp)	 #, n
	jg	L2	 #,
	movl	$1, %eax	 #, D.1493
	jmp	L3	 #
L2:
	movl	8(%ebp), %eax	 # n, tmp93
	subl	$1, %eax	 #, D.1493
	movl	%eax, (%esp)	 # D.1493,
	call	_fib	 #
	movl	%eax, %ebx	 #, D.1493
	movl	8(%ebp), %eax	 # n, tmp94
	subl	$2, %eax	 #, D.1493
	movl	%eax, (%esp)	 # D.1493,
	call	_fib	 #
	addl	%ebx, %eax	 # D.1493, D.1493
L3:
	addl	$20, %esp	 #,
	popl	%ebx	 #
	popl	%ebp	 #
	ret
	.ident	"GCC: (tdm-1) 5.1.0"

```

## 影片 

1. x86 Assembly #9 - Calling Conventions | cdecl
    * https://www.youtube.com/watch?v=frqPX7EHscM&list=PLan2CeTAw3pFOq5qc9urw8w7R-kvAT8Yb&index=9
2. x86 Assembly #10 - CALL | Call Instruction (讚!)
    * https://www.youtube.com/watch?v=xAK1pGf_QHw
3. x86 Assembly #11 - RET | Return Instruction (讚!)
    * https://www.youtube.com/watch?v=32zR57OTrAQ&list=PLan2CeTAw3pFOq5qc9urw8w7R-kvAT8Yb&index=11
4. x86 Assembly #27 - Local Variables and Arguments Passing | Part 2
    * https://www.youtube.com/watch?v=C8qASvZGHSo&list=PLan2CeTAw3pFOq5qc9urw8w7R-kvAT8Yb&index=27
