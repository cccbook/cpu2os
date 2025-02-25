set -x
gcc -m64 -w c4.c -o c4
./c4
# ./c4 -s hello.c
# ./c4 c4.c hello.c
# ./c4 c4.c c4.c hello.c