./c4 -o $1.bc $1.c
./bc2rv $1.bc > $1_bc.s
../rv/asm $1_bc.s -o $1_bc.o
../rv/vm $1_bc.
