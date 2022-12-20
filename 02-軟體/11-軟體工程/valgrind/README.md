# valgrind

* [C語言的記憶體洩漏(Memory Leak)偵測 - Valgrind](http://blog.yslin.tw/2014/03/c-valgrind.html)
* [ jserv / 課程討論區: 2021 年系統軟體課程 / J01:lab0](https://hackmd.io/@sysprog/linux2021-lab0)

## mem_test.c

```
$ wsl
wsl> gcc -g -o mem_test mem_test.c
wsl> valgrind --leak-check=full ./mem_test

Command 'valgrind' not found, but can be installed with:

sudo apt install valgrind

wsl> valgrind --leak-check=full ./mem_test
==2010== Memcheck, a memory error detector
==2010== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==2010== Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info    
==2010== Command: ./mem_test
==2010==
==2010== 
==2010== HEAP SUMMARY:
==2010==     in use at exit: 10 bytes in 1 blocks
==2010==   total heap usage: 1 allocs, 0 frees, 10 bytes allocated
==2010==
==2010== 10 bytes in 1 blocks are definitely lost in loss record 1 of 1
==2010==    at 0x483B7F3: malloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==2010==    by 0x10915E: func (mem_test.c:5)
==2010==    by 0x109172: main (mem_test.c:10)
==2010==
==2010== LEAK SUMMARY:
==2010==    definitely lost: 10 bytes in 1 blocks
==2010==    indirectly lost: 0 bytes in 0 blocks
==2010==      possibly lost: 0 bytes in 0 blocks
==2010==    still reachable: 0 bytes in 0 blocks
==2010==         suppressed: 0 bytes in 0 blocks
==2010==
==2010== For lists of detected and suppressed errors, rerun with: -s
==2010== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)   
```

## mem_access.c

```
wsl> gcc -g -o mem_access mem_access.c
wsl> valgrind --leak-check=full ./mem_access
==2016== Memcheck, a memory error detector
==2016== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.       
==2016== Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info    
==2016== Command: ./mem_access
==2016==
==2016== Invalid write of size 2
==2016==    at 0x1091AD: main (mem_access.c:9)
==2016==  Address 0x4a49044 is 0 bytes after a block of size 4 alloc'd
==2016==    at 0x483B7F3: malloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==2016==    by 0x10919E: main (mem_access.c:8)
==2016==
==2016== Invalid read of size 4
==2016==    at 0x1091D5: main (mem_access.c:14)
==2016==  Address 0x4a490a0 is 13 bytes after a block of size 3 alloc'd        
==2016==    at 0x483B7F3: malloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==2016==    by 0x1091C8: main (mem_access.c:13)
==2016==
==2016== Invalid read of size 4
==2016==    at 0x1091FA: main (mem_access.c:18)
==2016==  Address 0x4a49090 is 0 bytes inside a block of size 3 free'd
==2016==    at 0x483CA3F: free (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==2016==    by 0x1091F5: main (mem_access.c:15)
==2016==  Block was alloc'd at
==2016==    at 0x483B7F3: malloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==2016==    by 0x1091C8: main (mem_access.c:13)
==2016==
==2016== Invalid free() / delete / delete[] / realloc()
==2016==    at 0x483CA3F: free (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==2016==    by 0x10921A: main (mem_access.c:21)
==2016==  Address 0x4a49090 is 0 bytes inside a block of size 3 free'd
==2016==    at 0x483CA3F: free (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==2016==    by 0x1091F5: main (mem_access.c:15)
==2016==  Block was alloc'd at
==2016==    at 0x483B7F3: malloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==2016==    by 0x1091C8: main (mem_access.c:13)
==2016==
00==2016== 
==2016== HEAP SUMMARY:
==2016==     in use at exit: 0 bytes in 0 blocks
==2016==   total heap usage: 3 allocs, 4 frees, 1,031 bytes allocated
==2016==
==2016== All heap blocks were freed -- no leaks are possible
==2016==
==2016== For lists of detected and suppressed errors, rerun with: -s
==2016== ERROR SUMMARY: 5 errors from 4 contexts (suppressed: 0 from 0) 
```