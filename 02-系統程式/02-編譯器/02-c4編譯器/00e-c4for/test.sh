set -x
gcc -g -w c4.c -o c4
./c4 -s hello.c
./c4 -s test/fib.c
./c4 -s test/sum.c
./c4 -s test/for.c
# ./c4 -s test/for2.c
./c4 test/for.c
# ./c4 -u test/fib.c
# ./c4 test/fib.c
# ./c4 hello.c
./c4 c4.c hello.c
./c4 c4.c c4.c hello.c