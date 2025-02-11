./c4 -o $1.bc $1.c
./bc2c $1.bc > $1_bc.c
gcc -m32 $1_bc.c -o $1_bc.exe
gcc -m32 -S -O3 $1_bc.c -o $1_bc.s
./$1_bc.exe
