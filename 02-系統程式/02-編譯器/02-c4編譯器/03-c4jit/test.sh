set -x
gcc -w -m32 c4.c -o c4
./c4 hello.c
# ./c4 -s tests/fib.c
# ./c4 tests/fib.c 10
# ./c4 c4.c hello.c
# ./c4 c4.c c4.c hello.c
gcc -w -m32 c4x86.c -o c4x86
./c4x86 hello.c
./c4x86 tests/fib.c 10

