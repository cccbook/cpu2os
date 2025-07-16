set -x
gcc -D _MACOS -w c5.c -o c5
./c5 hello.c
./c5 -s test/fib.c
./c5 test/fib.c
# ./c5 c5.c hello.c
# ./c5 c5.c c5.c hello.c