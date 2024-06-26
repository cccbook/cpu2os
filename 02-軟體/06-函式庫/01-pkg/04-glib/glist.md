

## glib_list.c

```
wsl> export CFLAGS="`pkg-config --cflags glib-2.0` -g -Wall -std=gnu11 -O3"
export LDLIBS="`pkg-config --wsl> export LDLIBS="`pkg-config --libs   glib-2.0`"
wsl> make glib_list
cc -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -g -Wall -std=gnu11 -O3    glib_list.c  -lglib-2.0 -o glib_list
wsl> ./glib_list
a
b
c
```
