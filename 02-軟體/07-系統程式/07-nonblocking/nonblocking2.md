# nonblocking2.c

```
guest@localhost:~/cpu2os/02-軟體/07-系統程式/07-nonblocking$ gcc nonblocking2.
c -o nonblocking2
guest@localhost:~/cpu2os/02-軟體/07-系統程式/07-nonblocking$ ./nonblocking2
read /dev/tty: Resource temporarily unavailable
no input,buf is null
read /dev/tty: Resource temporarily unavailable
no input,buf is null
read /dev/tty: Resource temporarily unavailable
no input,buf is null
helloread /dev/tty: Resource temporarily unavailable
no input,buf is null

ret = 6, buf is hello

hi
ret = 3, buf is hi
lo

hread /dev/tty: Resource temporarily unavailable
no input,buf is null
elloread /dev/tty: Resource temporarily unavailable
no input,buf is null

ret = 6, buf is hello

cccccread /dev/tty: Resource temporarily unavailable
no input,buf is null
cccccccccccread /dev/tty: Resource temporarily unavailable
no input,buf is null

ret = 17, buf is cccccccccccccccc

read /dev/tty: Resource temporarily unavailable
no input,buf is null
read /dev/tty: Resource temporarily unavailable
no input,buf is null
read /dev/tty: Resource temporarily unavailable
no input,buf is null
```