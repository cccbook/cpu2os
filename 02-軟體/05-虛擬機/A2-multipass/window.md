# multipass windows

```
PS D:\pmedia> multipass shell primary
Welcome to Ubuntu 20.04.2 LTS (GNU/Linux 5.4.0-67-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

  System information as of Wed Mar 17 20:20:23 CST 2021

  System load:  0.82              Processes:             116
  Usage of /:   33.5% of 4.67GB   Users logged in:       1
  Memory usage: 62%               IPv4 address for eth0: 172.31.60.129
  Swap usage:   0%

 * Introducing self-healing high availability clusters in MicroK8s.
   Simple, hardened, Kubernetes for production, from RaspberryPi to DC.

     https://microk8s.io/high-availability

1 update can be installed immediately.
0 of these updates are security updates.
To see these additional updates run: apt list --upgradable


Last login: Wed Mar 17 20:20:15 2021 from 172.31.48.1
ubuntu@primary:~$ ls
Home  ccc  snap  test.md
ubuntu@primary:~$ gcc
gcc: fatal error: no input files
compilation terminated.
ubuntu@primary:~$ ls
Home  ccc  snap  test.md
ubuntu@primary:~$ cd ccc
ubuntu@primary:~/ccc$ ls
sp
ubuntu@primary:~/ccc$ cd sp
ubuntu@primary:~/ccc/sp$ ls
01-sp        05-obj    09-linux    C1-c4     C5-riscv-emu  _doc      系統程式課程.md
02-gcc       06-vm     10-riscv    C2-c4rv   C6-simplefs   _more
03-compiler  07-lib    A1-se       C3-rubi   C7-sehttpd    clear.sh
04-asm       08-posix  A2-arduino  C4-shecc  _c            map.md
ubuntu@primary:~/ccc/sp$ cd 09-linux/
ubuntu@primary:~/ccc/sp/09-linux$ ls
01-c  02-asm  03-elf  04-dll
ubuntu@primary:~/ccc/sp/09-linux$ cd 02-asm/
ubuntu@primary:~/ccc/sp/09-linux/02-asm$ ls
00-preface  02-hola  04-maxofthree  06-power  08-average    Linux上的x86組合語言設計.md  myMacro.s
01-hello    03-fib   05-echo        07-sum    09-factorial  backup
ubuntu@primary:~/ccc/sp/09-linux/02-asm$ cd 01-hello/
ubuntu@primary:~/ccc/sp/09-linux/02-asm/01-hello$ l
README.md  hello*  hello.o  hello.s  helloMacro*  helloMacro.o  helloMacro.s
ubuntu@primary:~/ccc/sp/09-linux/02-asm/01-hello$ ls
README.md  hello  hello.o  hello.s  helloMacro  helloMacro.o  helloMacro.s
ubuntu@primary:~/ccc/sp/09-linux/02-asm/01-hello$ ./hello
Hello, world
```
