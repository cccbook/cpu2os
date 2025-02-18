# minix

* http://www.minix3.org/

## 下載

* 下載點 -- https://wiki.minix3.org/doku.php?id=www:download:start

## 在 qemu 中執行

* 參考 -- https://wiki.minix3.org/doku.php?id=usersguide:runningonqemu

```
user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/06-vm/04-minix (master)
$ cd iso

user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/06-vm/04-minix/iso (master)
$ qemu-system-x86_64 -net user -net nic -m 256 -cdrom minix.iso -hda minix.img -boot d
WARNING: Image format was not specified for 'minix.img' and probing guessed raw.
         Automatically detecting the format is dangerous for raw images, write operations on block 0 will be restricted.
         Specify the 'raw' format explicitly to remove the restrictions.
```
