gcc `pkg-config --cflags glib-2.0` $1.c `pkg-config --libs   glib-2.0` -o $1
./$1
