gcc `pkg-config --cflags gmp` $1.c `pkg-config --libs gmp` -o $1
./$1