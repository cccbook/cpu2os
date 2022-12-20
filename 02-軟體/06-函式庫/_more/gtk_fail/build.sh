gcc  -I/mingw64/include/gtk-4.0 -I/mingw64/include -I/mingw64/include/glib-2.0 -I/mingw64/lib/glib-2.0/include -I/mingw64/include $1.c -L/mingw64/lib -lgtk-4 -o $1
./$1
