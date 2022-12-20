# shell pipe

```
mac020:03-pipe mac020$ ls -l
total 48
-rw-r--r--  1 mac020  staff   282  5 25 10:00 README.md
-rwxr-xr-x  1 mac020  staff  8916  5 25 09:59 pipe0
-rw-r--r--  1 mac020  staff  1142  5 25 09:57 pipe0.c
-rw-r--r--  1 mac020  staff   146  3  3 10:11 pipe1.c
mac020:03-pipe mac020$ ls -l > ls.out
mac020:03-pipe mac020$ cat ls.out
total 48
-rw-r--r--  1 mac020  staff   282  5 25 10:00 README.md
-rw-r--r--  1 mac020  staff     0  5 25 10:03 ls.out
-rwxr-xr-x  1 mac020  staff  8916  5 25 09:59 pipe0
-rw-r--r--  1 mac020  staff  1142  5 25 09:57 pipe0.c
-rw-r--r--  1 mac020  staff   146  3  3 10:11 pipe1.c
mac020:03-pipe mac020$ ps
  PID TTY           TIME CMD
 9065 ttys000    0:00.02 /bin/bash -l
13086 ttys001    0:00.02 /bin/bash -l
17498 ttys002    0:00.01 /bin/bash -l
24191 ttys003    0:00.02 /bin/bash -l
26804 ttys004    0:00.02 /bin/bash -l
mac020:03-pipe mac020$ cat README.md | wc
      16      31     282
mac020:03-pipe mac020$ wc README.md
      16      31     282 README.md
```