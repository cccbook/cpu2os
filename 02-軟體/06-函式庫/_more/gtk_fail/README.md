# gtk

* https://docs.gtk.org/gtk4/getting_started.html
* https://openhome.cc/Gossip/GTKGossip/FirstGTKWindows.html


## fail

```
$ ./build.sh gtk2
In file included from /mingw64/include/glib-2.0/glib/galloca.h:32,
                 from /mingw64/include/glib-2.0/glib.h:30,
                 from /mingw64/include/gtk-4.0/gtk/css/gtkcss.h:30,
                 from /mingw64/include/gtk-4.0/gtk/gtk.h:30,
                 from gtk2.c:1:
/mingw64/include/glib-2.0/glib/gtypes.h:32:10: fatal error: glibconfig.h: No such file or directory
   32 | #include <glibconfig.h>
      |          ^~~~~~~~~~~~~~
compilation terminated.
./build.sh: line 2: ./gtk2: No such file or directory

```