# mmap

https://man7.org/linux/man-pages/man2/mmap.2.html

```
EXAMPLES 
       The following program prints part of the file specified in its
       first command-line argument to standard output.  The range of
       bytes to be printed is specified via offset and length values in
       the second and third command-line arguments.  The program creates
       a memory mapping of the required pages of the file and then uses
       write(2) to output the desired bytes.
```

## run

```
$ gcc readPrint.c -o readPrint
$ ./readPrint readPrint.c 10 50
...
```