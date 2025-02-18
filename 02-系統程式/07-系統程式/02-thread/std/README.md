# C11

## atomic1.c

```
wsl> gcc -ggdb3 -O0 -std=c99 -Wall -Wextra -pedantic -o atomic1 atomic1.c -pthread
wsl> ./atomic1
The atomic counter is 10000
The non-atomic counter is 10000
```

## noraceAtomic1.c

```
wsl> gcc -ggdb3 -O0 -std=c99 -Wall -Wextra -pedantic -o noraceAtomic noraceAtomic.c -pthread
wsl> ./noraceAtomic
counter=0
```