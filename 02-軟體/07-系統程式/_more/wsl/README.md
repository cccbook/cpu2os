# Window Subsystem for Linux

https://docs.microsoft.com/zh-tw/windows/wsl/install-win10


```
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username: ccckmit
New password:
Retype new password:
passwd: password updated successfully
Installation successful!
To run a command as administrator (user "root"), use "sudo <command>".
See "man sudo_root" for details.

Welcome to Ubuntu 20.04.1 LTS (GNU/Linux 4.4.0-18362-Microsoft x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

  System information as of Sat Oct 31 07:46:35 CST 2020

  System load:    0.52      Users logged in:       0
  Usage of /home: unknown   IPv4 address for eth1: 192.168.42.177
  Memory usage:   75%       IPv4 address for eth2: 192.168.9.1
  Swap usage:     3%        IPv4 address for eth3: 192.168.205.1
  Processes:      7

1 update can be installed immediately.
0 of these updates are security updates.
To see these additional updates run: apt list --upgradable


The list of available updates is more than a week old.
To check for new updates run: sudo apt update


This message is shown once once a day. To disable it please create the
/home/ccckmit/.hushlogin file.
ccckmit@DESKTOP-96FRN6B:~$ pwd
/home/ccckmit
ccckmit@DESKTOP-96FRN6B:~$ ls
ccckmit@DESKTOP-96FRN6B:~$ ls /
bin   dev  home  lib    lib64   media  opt   root  sbin  srv  tmp  var
boot  etc  init  lib32  libx32  mnt    proc  run   snap  sys  usr
ccckmit@DESKTOP-96FRN6B:~$ ls
ccckmit@DESKTOP-96FRN6B:~$ vim
ccckmit@DESKTOP-96FRN6B:~$ gcc

Command 'gcc' not found, but can be installed with:

sudo apt install gcc

ccckmit@DESKTOP-96FRN6B:~$ sudo apt install gcc
[sudo] password for ccckmit:
Reading package lists... Done
Building dependency tree
Reading state information... Done
The following additional packages will be installed:
  binutils binutils-common binutils-x86-64-linux-gnu cpp cpp-9 gcc-9 gcc-9-base libasan5 libatomic1 libbinutils
  libc-dev-bin libc6-dev libcc1-0 libcrypt-dev libctf-nobfd0 libctf0 libgcc-9-dev libgomp1 libisl22 libitm1 liblsan0
  libmpc3 libquadmath0 libtsan0 libubsan1 linux-libc-dev manpages-dev
Suggested packages:
  binutils-doc cpp-doc gcc-9-locales gcc-multilib make autoconf automake libtool flex bison gdb gcc-doc gcc-9-multilib
  gcc-9-doc glibc-doc
The following NEW packages will be installed:
  binutils binutils-common binutils-x86-64-linux-gnu cpp cpp-9 gcc gcc-9 gcc-9-base libasan5 libatomic1 libbinutils
  libc-dev-bin libc6-dev libcc1-0 libcrypt-dev libctf-nobfd0 libctf0 libgcc-9-dev libgomp1 libisl22 libitm1 liblsan0
  libmpc3 libquadmath0 libtsan0 libubsan1 linux-libc-dev manpages-dev
0 upgraded, 28 newly installed, 0 to remove and 0 not upgraded.
Need to get 28.5 MB of archives.
After this operation, 123 MB of additional disk space will be used.
Do you want to continue? [Y/n] y
Get:1 http://archive.ubuntu.com/ubuntu focal/main amd64 binutils-common amd64 2.34-6ubuntu1 [207 kB]
Get:2 http://archive.ubuntu.com/ubuntu focal/main amd64 libbinutils amd64 2.34-6ubuntu1 [474 kB]
Get:3 http://archive.ubuntu.com/ubuntu focal/main amd64 libctf-nobfd0 amd64 2.34-6ubuntu1 [47.0 kB]
Get:4 http://archive.ubuntu.com/ubuntu focal/main amd64 libctf0 amd64 2.34-6ubuntu1 [46.6 kB]
Get:5 http://archive.ubuntu.com/ubuntu focal/main amd64 binutils-x86-64-linux-gnu amd64 2.34-6ubuntu1 [1614 kB]
Get:6 http://archive.ubuntu.com/ubuntu focal/main amd64 binutils amd64 2.34-6ubuntu1 [3376 B]
Get:7 http://archive.ubuntu.com/ubuntu focal/main amd64 gcc-9-base amd64 9.3.0-10ubuntu2 [19.3 kB]
Get:8 http://archive.ubuntu.com/ubuntu focal/main amd64 libisl22 amd64 0.22.1-1 [592 kB]
Get:9 http://archive.ubuntu.com/ubuntu focal/main amd64 libmpc3 amd64 1.1.0-1 [40.8 kB]
Get:10 http://archive.ubuntu.com/ubuntu focal/main amd64 cpp-9 amd64 9.3.0-10ubuntu2 [7491 kB]
Get:11 http://archive.ubuntu.com/ubuntu focal/main amd64 cpp amd64 4:9.3.0-1ubuntu2 [27.6 kB]
Get:12 http://archive.ubuntu.com/ubuntu focal/main amd64 libcc1-0 amd64 10-20200411-0ubuntu1 [41.1 kB]
Get:13 http://archive.ubuntu.com/ubuntu focal/main amd64 libgomp1 amd64 10-20200411-0ubuntu1 [101 kB]
Get:14 http://archive.ubuntu.com/ubuntu focal/main amd64 libitm1 amd64 10-20200411-0ubuntu1 [26.3 kB]
Get:15 http://archive.ubuntu.com/ubuntu focal/main amd64 libatomic1 amd64 10-20200411-0ubuntu1 [9284 B]
Get:16 http://archive.ubuntu.com/ubuntu focal/main amd64 libasan5 amd64 9.3.0-10ubuntu2 [395 kB]
Get:17 http://archive.ubuntu.com/ubuntu focal/main amd64 liblsan0 amd64 10-20200411-0ubuntu1 [144 kB]
Get:18 http://archive.ubuntu.com/ubuntu focal/main amd64 libtsan0 amd64 10-20200411-0ubuntu1 [319 kB]
Get:19 http://archive.ubuntu.com/ubuntu focal/main amd64 libubsan1 amd64 10-20200411-0ubuntu1 [136 kB]
Get:20 http://archive.ubuntu.com/ubuntu focal/main amd64 libquadmath0 amd64 10-20200411-0ubuntu1 [146 kB]
Get:21 http://archive.ubuntu.com/ubuntu focal/main amd64 libgcc-9-dev amd64 9.3.0-10ubuntu2 [2359 kB]
Get:22 http://archive.ubuntu.com/ubuntu focal/main amd64 gcc-9 amd64 9.3.0-10ubuntu2 [8234 kB]
Get:23 http://archive.ubuntu.com/ubuntu focal/main amd64 gcc amd64 4:9.3.0-1ubuntu2 [5208 B]
Get:24 http://archive.ubuntu.com/ubuntu focal/main amd64 libc-dev-bin amd64 2.31-0ubuntu9 [71.8 kB]
Ign:25 http://archive.ubuntu.com/ubuntu focal-updates/main amd64 linux-libc-dev amd64 5.4.0-42.46
Get:26 http://archive.ubuntu.com/ubuntu focal/main amd64 libcrypt-dev amd64 1:4.4.10-10ubuntu4 [104 kB]
Get:27 http://archive.ubuntu.com/ubuntu focal/main amd64 libc6-dev amd64 2.31-0ubuntu9 [2520 kB]
Err:25 http://security.ubuntu.com/ubuntu focal-updates/main amd64 linux-libc-dev amd64 5.4.0-42.46
  404  Not Found [IP: 91.189.88.142 80]
Get:28 http://archive.ubuntu.com/ubuntu focal/main amd64 manpages-dev all 5.05-1 [2266 kB]
Fetched 27.4 MB in 2min 13s (207 kB/s)
E: Failed to fetch http://security.ubuntu.com/ubuntu/pool/main/l/linux/linux-libc-dev_5.4.0-42.46_amd64.deb  404  Not Found [IP: 91.189.88.142 80]
E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?

```