# termios

## password.c

```
mac020:10-terminal mac020$ gcc password.c -o password -lcurses
mac020:10-terminal mac020$ ./password
Enter password: 
You Entered dsajkfd
```

## sizeterm.c

```
mac020:10-terminal mac020$ gcc sizeterm.c -o sizeterm -lcurses
mac020:10-terminal mac020$ ./sizeterm
nrows=12 ncolumns=87
// 調大視窗後
mac020:10-terminal mac020$ ./sizeterm
nrows=16 ncolumns=87
```