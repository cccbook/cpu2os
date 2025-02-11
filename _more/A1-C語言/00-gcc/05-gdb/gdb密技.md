

* https://www.facebook.com/cih.tw/posts/pfbid0eXHjKcy444XUufnULFtGceSthM1wcShF4QaqwCDnsxqGs1nbMUNNKsejfeNdDSiNl

```
gcc -g program.c -o program
gdb -x trace.gdb ./program
------trace.gdb-----
break main
run
while 1
step
frame
end
```
