set -x
gcc -w c4.c -o c4
#./c4 -s -t test/var.c
#./c4 test/fib.c
#./c4 -s -t test/sum.c
./c4 hello.c
# ./c4 c4.c hello.c
#./c4 c4.c c4.c hello.c