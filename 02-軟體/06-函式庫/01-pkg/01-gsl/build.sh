gcc `pkg-config --cflags gsl` $1.c `pkg-config --libs gsl` -o $1
./$1