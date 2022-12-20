# user level thread

* [處理高併發服務的用戶級線程 (User Level Thread)](https://hkt999.medium.com/%E5%AF%A6%E4%BD%9C%E7%94%A8%E6%88%B6%E7%B4%9A%E7%B7%9A%E7%A8%8B-user-level-thread-5424afd61274)
    * https://github.com/hkt999/user-level-thread

## run

```
guest@localhost:~/sp/09-linux/01-c/03-user-level-thread/src$ make
gcc -g main.c thread.c platform.c -o test -D_XOPEN_SOURCE
thread.c: In function ‘thread_yield’:
thread.c:366:4: warning: implicit declaration of function ‘usleep’; did you mean ‘sleep’? [-Wimplicit-function-declaration]
    usleep(10000); /* 10ms */
    ^~~~~~
    sleep
platform.c: In function ‘platform_create_context’:
platform.c:62:2: warning: implicit declaration of function ‘bzero’ [-Wimplicit-function-declaration]
  bzero(th->th_context.th_stack, stacksize);
  ^~~~~
platform.c:62:2: warning: incompatible implicit declaration of built-in function ‘bzero’
guest@localhost:~/sp/09-linux/01-c/03-user-level-thread/src$ ./test
worker (thread1) -- 0
worker (thread2) -- 0
worker (thread3) -- 0
worker (thread1) -- 1
worker (thread2) -- 1
worker (thread1) -- 2
worker (thread3) -- 1
worker (thread1) -- 3
worker (thread2) -- 2
worker (thread1) -- 4
worker (thread1) -- 5
worker (thread2) -- 3
worker (thread3) -- 2
worker (thread1) -- 6
worker (thread2) -- 4
worker (thread1) -- 7
worker (thread3) -- 3
worker (thread1) -- 8
worker (thread2) -- 5
worker (thread1) -- 9
end of workder (thread1)
worker (thread2) -- 6
worker (thread3) -- 4
worker (thread2) -- 7
worker (thread3) -- 5
worker (thread2) -- 8
worker (thread2) -- 9
worker (thread3) -- 6
---- thread information ----
thread ready count: 1
thread sleep count: 2
[thread3]                                       6(C)     6(R),     0(S),     7(SL) time(5638)
[thread2]                                       9(C)     9(R),     0(S),    10(SL) time(5065)
[main thread]                                   1(C)     1(R),     0(S),     1(SL) time(0)
---------------------------
end of workder (thread2)
worker (thread3) -- 7
worker (thread3) -- 8
worker (thread3) -- 9
end of workder (thread3)
---- thread information ----
thread ready count: 1
thread sleep count: 0
[thread3]                                       4(C)    10(R),     0(S),    10(SL) time(0)
[main thread]                                   1(C)     2(R),     0(S),     2(SL) time(0)
---------------------------
---- thread information ----
thread ready count: 1
thread sleep count: 0
[main thread]                                   1(C)     3(R),     0(S),     3(SL) time(0)
---------------------------
^C
```