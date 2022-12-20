# 用 make 建置執行檔 

```
wsl> make clean
rm -f *.o *.exe run
wsl> make
gcc -std=c99 -O0 sum.c main.c -o run
wsl> ./run
sum(10)=55
```
