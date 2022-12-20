# multipass

* https://github.com/canonical/multipass/issues/1447

start failed: The following errors occurred:                                    
Instance stopped while starting
logout                                                                          
Saving session...
...copying shared history...
...saving history...truncating history files...
...completed.

[程序完成]


解決方法

I solved the problem by disabling internet sharing.
For days I was digging the problem to the point I almost gave up. Almost turned to other VM solutions. Internet sharing was turned on because of crappy Wi-Fi in my room and I got a ethernet cable to my Mac.
It appears internet sharing changes some settings that Multipass did not expect. I wish this piece of info helps tho.

## 成功

```
(base) csienqu-teacher:ccc csienqu$ multipass delete primary               
(base) csienqu-teacher:ccc csienqu$ multipass list
Name                    State             IPv4             Image
primary                 Deleted           --               Not Available
(base) csienqu-teacher:ccc csienqu$ multipass list
Name                    State             IPv4             Image
primary                 Deleted           --               Not Available
(base) csienqu-teacher:ccc csienqu$ ultipass launch -n ubuntu-lts -c 4 -m 4G -d 40G
bash: ultipass: command not found
(base) csienqu-teacher:ccc csienqu$ multipass launch -n ubuntu-lts -c 4 -m 4G -d 40G
Launched: ubuntu-lts                                                            
(base) csienqu-teacher:ccc csienqu$ multipass listName                    State             IPv4             Image
primary                 Deleted           --               Not Available
ubuntu-lts              Running           192.168.64.2     Ubuntu 20.04 LTS
(base) csienqu-teacher:ccc csienqu$ multipass shell ubuntu-lts
Welcome to Ubuntu 20.04.2 LTS (GNU/Linux 5.4.0-67-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

  System information as of Thu Mar 18 08:54:24 CST 2021

  System load:  0.14              Processes:               159
  Usage of /:   3.3% of 38.60GB   Users logged in:         0
  Memory usage: 5%                IPv4 address for enp0s2: 192.168.64.2
  Swap usage:   0%


1 update can be installed immediately.
0 of these updates are security updates.
To see these additional updates run: apt list --upgradable


To run a command as administrator (user "root"), use "sudo <command>".
See "man sudo_root" for details.

ubuntu@ubuntu-lts:~$ ls
ubuntu@ubuntu-lts:~$ gcc

Command 'gcc' not found, but can be installed with:

sudo apt install gcc

ubuntu@ubuntu-lts:~$ sudo apt install gcc
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following additional packages will be installed:
  binutils binutils-common binutils-x86-64-linux-gnu cpp cpp-9 gcc-9
  gcc-9-base libasan5 libatomic1 libbinutils libc-dev-bin libc6-dev
  libcc1-0 libcrypt-dev libctf-nobfd0 libctf0 libgcc-9-dev libgomp1
  libisl22 libitm1 liblsan0 libmpc3 libquadmath0 libtsan0 libubsan1
  linux-libc-dev manpages-dev
Suggested packages:
  binutils-doc cpp-doc gcc-9-locales gcc-multilib make autoconf automake
  libtool flex bison gdb gcc-doc gcc-9-multilib gcc-9-doc glibc-doc
The following NEW packages will be installed:
  binutils binutils-common binutils-x86-64-linux-gnu cpp cpp-9 gcc gcc-9
  gcc-9-base libasan5 libatomic1 libbinutils libc-dev-bin libc6-dev
  libcc1-0 libcrypt-dev libctf-nobfd0 libctf0 libgcc-9-dev libgomp1
  libisl22 libitm1 liblsan0 libmpc3 libquadmath0 libtsan0 libubsan1
  linux-libc-dev manpages-dev
0 upgraded, 28 newly installed, 0 to remove and 0 not upgraded.
Need to get 28.6 MB of archives.
After this operation, 123 MB of additional disk space will be used.
Do you want to continue? [Y/n] 
...

ubuntu@ubuntu-lts:~$ gcc
gcc: fatal error: no input files
compilation terminated.
```
