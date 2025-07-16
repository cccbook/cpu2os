# linux

## georgeMary.c

```
guest@localhost:~/code/sp/os/01-thread$ gcc georgeMary.c -lpthread -o georgeMary
guest@localhost:~/code/sp/os/01-thread$ ./georgeMary
----------------
Mary
George
----------------
George
----------------
Mary
George
----------------
George
----------------
Mary
George
```

## Others

```
guest@localhost:~/code/sp/os/01-thread$ gcc race.c -lpthread -o race
guest@localhost:~/code/sp/os/01-thread$ ./race
counter=-2338459
guest@localhost:~/code/sp/os/01-thread$ gcc norace.c -lpthread -o norace
guest@localhost:~/code/sp/os/01-thread$ ./norace
counter=0
guest@localhost:~/code/sp/os/01-thread$ gcc deadlock.c -lpthread -o deadlock
guest@localhost:~/code/sp/os/01-thread$ ./deadlock
B lock y
A lock x

```