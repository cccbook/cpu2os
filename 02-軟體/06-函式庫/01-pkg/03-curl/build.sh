gcc `pkg-config --cflags libcurl` $1.c `pkg-config --libs libcurl` -o $1
./$1