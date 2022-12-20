# terminal

isatty(fd): 檢查檔案描述子 fd 是否為一終端機。

## termios

使用 termios.h 必須在 gcc 加入 -lcurses 連結 libcurses.a 函式庫。

## stty -a

顯示 termios 的設定

```
guest@localhost:~$ stty -a
speed 38400 baud; rows 5; columns 87; line = 0;
intr = ^C; quit = ^\; erase = ^?; kill = ^U; eof = ^D; eol = M-^?; eol2 = M-^?;
swtch = <undef>; start = ^Q; stop = ^S; susp = ^Z; rprnt = ^R; werase = ^W; lnext = ^V;
discard = ^O; min = 1; time = 0;
-parenb -parodd -cmspar cs8 -hupcl -cstopb cread -clocal -crtscts
-ignbrk -brkint -ignpar -parmrk -inpck -istrip -inlcr -igncr icrnl ixon -ixoff -iuclc
ixany imaxbel iutf8
opost -olcuc -ocrnl onlcr -onocr -onlret -ofill -ofdel nl0 cr0 tab0 bs0 vt0 ff0
isig icanon iexten echo echoe echok -echonl -noflsh -xcase -tostop -echoprt echoctl
echoke -flusho -extproc
```

## $TERM

```
mac020:10-terminal mac020$ echo $TERM
xterm-256color
```
