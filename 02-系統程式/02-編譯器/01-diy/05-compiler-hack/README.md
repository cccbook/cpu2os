# Compiler-Hack

可以產生 HackCPU 的組合語言

```
user@DESKTOP-96FRN6B MINGW64 /d/ccc109/sp/02-compiler/05-compiler-hack (master)
$ make
gcc -std=c99 -O0 ir.c lexer.c compiler.c main.c -o compiler

user@DESKTOP-96FRN6B MINGW64 /d/ccc109/sp/02-compiler/05-compiler-hack (master)
$ ./compiler test/sum.c

s=0;
i=1;
while (i < 10) {
  s = s + i;    
  i = i + 1;    
}


========== lex ==============
token=s
token==
token=0
token=;
token=i
token==
token=1
token=;
token=while
token=(
token=i
token=<
token=10
token=)
token={
token=s
token==
token=s
token=+
token=i
token=;
token=i
token==
token=i
token=+
token=1
token=;
token=}
========== dump ==============
0:s
1:=
2:0
3:;
4:i
5:=
6:1
7:;
8:while
9:(
10:i
11:<
12:10
13:)
14:{
15:s
16:=
17:s
18:+
19:i
20:;
21:i
22:=
23:i
24:+
25:1
26:;
27:}
============ parse =============
============ irDump ============
t1 = 0
s = t1
t2 = 1
i = t2
(L1)
t3 = i
t4 = 10
t5 = t3 < t4
if t5 goto L1
t6 = s
t7 = i
t8 = t6 + t7
s = t8
t9 = i
t10 = 1
t11 = t9 + t10
i = t11
goto L1
(L2)
============ irToHack ==========
# init NF (negative flag mask)
@32767
D=A
@NF
M=D
# t1 = 0
@0
D=A
@t1
M=D

# s = t1
@t1
D=M
@s
M=D

# t2 = 1
@1
D=A
@t2
M=D

# i = t2
@t2
D=M
@i
M=D

# (L1)
(L1)
# t3 = i
@i
D=M
@t3
M=D

# t4 = 10
@10
D=A
@t4
M=D

# t5 = t3 < t4
@t3
D=M
@t4
D=D-M
@NF
D=D&M

@t5
M=D

# if t5 goto L1
@t5
D=M
@L1
D;JEQ

# t6 = s
@s
D=M
@t6
M=D

# t7 = i
@i
D=M
@t7
M=D

# t8 = t6 + t7
@t6
D=M
@t7
+
@t8
M=D

# s = t8
@t8
D=M
@s
M=D

# t9 = i
@i
D=M
@t9
M=D

# t10 = 1
@1
D=A
@t10
M=D

# t11 = t9 + t10
@t9
D=M
@t10
+
@t11
M=D

# i = t11
@t11
D=M
@i
M=D

# goto L1
@L1
0;JMP

# (L2)
(L2)
```