set -x
gcc -w -g c4.c -o c4
./c4 -s hello.c0
# ./c4
# ./c4 -s hello.c
# ./c4 c4.c hello.c
# ./c4 c4.c c4.c hello.c
