# ccc 的 riscv-emu 筆記

* [2019q1 Homework4 (riscv)](https://hackmd.io/@ZMcDFsJ8Qrm7NTG2hwqYXA/rkjbLGmtN?type=view)

> virtio 是一個在模擬器的環境下半虛擬化的技術，針對 I/O 虛擬化的一個通用介面框架。

* [Virtio: An I/O virtualization framework for Linux](https://developer.ibm.com/articles/l-virtio/)

## Install

```
$ sudo apt-get install libcurl4-openssl-dev
$ sudo apt-get install libsdl2-dev
```

## build

```
root@localhost:~/ccc/riscv-emu# make
gcc -O2 -Wall -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -D_GNU_SOURCE -DCONFIG_VERSION=\"2019-02-10\" 
-DCONFIG_SLIRP -DCONFIG_FS_NET -DCONFIG_SDL -DCONFIG_RISCV_MAX_XLEN=128 -DCONFIG_COMPRESSED_INITRAMFS -c -o sdl.o sdl.c
gcc -O2 -Wall -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -D_GNU_SOURCE -DCONFIG_VERSION=\"2019-02-10\" 
-DCONFIG_SLIRP -DCONFIG_FS_NET -DCONFIG_SDL -DCONFIG_RISCV_MAX_XLEN=128 -DCONFIG_COMPRESSED_INITRAMFS -c -o riscv_machine.o riscv_machine.c
gcc -O2 -Wall -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -D_GNU_SOURCE -DCONFIG_VERSION=\"2019-02-10\" 
-DCONFIG_SLIRP -DCONFIG_FS_NET -DCONFIG_SDL -DCONFIG_RISCV_MAX_XLEN=128 -DCONFIG_COMPRESSED_INITRAMFS -c -o softfp.o softfp.c
gcc -O2 -Wall -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -D_GNU_SOURCE -DCONFIG_VERSION=\"2019-02-10\" 
-DCONFIG_SLIRP -DCONFIG_FS_NET -DCONFIG_SDL -DCONFIG_RISCV_MAX_XLEN=128 -DCONFIG_COMPRESSED_INITRAMFS -DMAX_XLEN=32 -c -o riscv_cpu32.o riscv_cpu.c
gcc -O2 -Wall -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -D_GNU_SOURCE -DCONFIG_VERSION=\"2019-02-10\" 
-DCONFIG_SLIRP -DCONFIG_FS_NET -DCONFIG_SDL -DCONFIG_RISCV_MAX_XLEN=128 -DCONFIG_COMPRESSED_INITRAMFS -DMAX_XLEN=64 -c -o riscv_cpu64.o riscv_cpu.c
gcc -O2 -Wall -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -D_GNU_SOURCE -DCONFIG_VERSION=\"2019-02-10\" 
-DCONFIG_SLIRP -DCONFIG_FS_NET -DCONFIG_SDL -DCONFIG_RISCV_MAX_XLEN=128 -DCONFIG_COMPRESSED_INITRAMFS -DMAX_XLEN=128 -c -o riscv_cpu128.o riscv_cpu.c
gcc -O2 -Wall -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -D_GNU_SOURCE -DCONFIG_VERSION=\"2019-02-10\" 
-DCONFIG_SLIRP -DCONFIG_FS_NET -DCONFIG_SDL -DCONFIG_RISCV_MAX_XLEN=128 -DCONFIG_COMPRESSED_INITRAMFS -c -o compress.o compress.c
gcc -o temu virtio.o pci.o fs.o cutils.o iomem.o simplefb.o json.o machine.o temu.o elf.o slirp/bootp.o slirp/ip_icmp.o slirp/mbuf.o slirp/slirp.o slirp/tcp_output.o slirp/cksum.o slirp/ip_input.o slirp/misc.o slirp/socket.o slirp/tcp_subr.o slirp/udp.o slirp/if.o slirp/ip_output.o slirp/sbuf.o slirp/tcp_input.o slirp/tcp_timer.o fs_disk.o fs_net.o fs_wget.o fs_utils.o block_net.o sdl.o riscv_machine.o softfp.o riscv_cpu32.o riscv_cpu64.o riscv_cpu128.o compress.o  -lz -lrt -lcurl -lcrypto -lSDL2
gcc -O2 -Wall -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -D_GNU_SOURCE -DCONFIG_VERSION=\"2019-02-10\" 
-DCONFIG_SLIRP -DCONFIG_FS_NET -DCONFIG_SDL -DCONFIG_RISCV_MAX_XLEN=128 -DCONFIG_COMPRESSED_INITRAMFS -c -o build_filelist.o build_filelist.c
gcc -o build_filelist build_filelist.o fs_utils.o cutils.o  -lz -lm
gcc -O2 -Wall -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -MMD -D_GNU_SOURCE -DCONFIG_VERSION=\"2019-02-10\" 
-DCONFIG_SLIRP -DCONFIG_FS_NET -DCONFIG_SDL -DCONFIG_RISCV_MAX_XLEN=128 -DCONFIG_COMPRESSED_INITRAMFS -c -o splitimg.o splitimg.c
gcc -o splitimg splitimg.o  -lz
root@localhost:~/ccc/riscv-emu# ls
aes.c             elf.c       fs_utils.o  machine.d       riscv_cpu_fp_template.h  softfp.h
aes.h             elf.d       fs_wget.c   machine.h       riscv_cpu.h              softfp.o
block_net.c       elf.h       fs_wget.d   machine.o       riscv_cpu_priv.h         softfp_template.h
block_net.d       elf.o       fs_wget.h   Makefile        riscv_cpu_template.h     softfp_template_icvt.h      
block_net.o       fbuf.h      fs_wget.o   Makefile.js     riscv_machine.c          splitimg
build_filelist    fs.c        iomem.c     netinit.sh      riscv_machine.d          splitimg.c
build_filelist.c  fs.d        iomem.d     pci.c           riscv_machine.o          splitimg.d
build_filelist.d  fs_disk.c   iomem.h     pci.d           sdl.c                    splitimg.o
build_filelist.o  fs_disk.d   iomem.o     pci.h           sdl.d                    temu
Changelog         fs_disk.o   js          pci.o           sdl.o                    temu.c
compress.c        fs.h        jsemu.c     README.md       sha256.c                 temu.d
compress.d        fs_net.c    json.c      riscv_cpu128.d  sha256.h                 temu.o
compress.h        fs_net.d    json.d      riscv_cpu128.o  simplefb.c               VERSION
compress.o        fs_net.o    json.h      riscv_cpu32.d   simplefb.d               virtio.c
cutils.c          fs.o        json.o      riscv_cpu32.o   simplefb.o               virtio.d
cutils.d          fs_utils.c  LICENSE     riscv_cpu64.d   slirp                    virtio.h
cutils.h          fs_utils.d  list.h      riscv_cpu64.o   softfp.c                 virtio.o
cutils.o          fs_utils.h  machine.c   riscv_cpu.c     softfp.d
root@localhost:~/ccc/riscv-emu#
root@localhost:~/ccc/riscv-emu# ./temu https://bellard.org/jslinux/buildroot-riscv64.cfg

Welcome to JS/Linux (riscv64)

Use 'vflogin username' to connect to your account.
You can create a new account at https://vfsync.org/signup .
Use 'export_file filename' to export a file to your computer.
Imported files are written to the home directory.

[root@localhost ~]# ls
bench.py       hello.c        hello.js       readme.txt     rv128test.bin
[root@localhost ~]# gcc hello.c
[root@localhost ~]# ls
a.out          bench.py       hello.c        hello.js       readme.txt     rv128test.bin
[root@localhost ~]# ./a.out
Hello World
```