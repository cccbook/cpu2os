# blocking vs non-blocking

* https://programming.vip/docs/linux-blocking-and-non-blocking.html (讚)
* [非同步程式設計和 non-blocking IO](https://medium.com/@fcamel/%E9%9D%9E%E5%90%8C%E6%AD%A5%E7%A8%8B%E5%BC%8F%E8%A8%AD%E8%A8%88%E5%92%8C-non-blocking-io-a43881081aac)
* [淺談I/O Model](https://medium.com/@clu1022/%E6%B7%BA%E8%AB%87i-o-model-32da09c619e6) (讚)
* [Linux kernel模組的開發系列 第 26 篇 Blocking 與 Non-blocking I/O](https://ithelp.ithome.com.tw/articles/10161404)

## blocking1.c

```
$ gcc blocking1.c -o blocking1
$ ./blocking1
hahaha
buf is hahaha

test
hello
buf is hello


test
afljdasfl;j
buf is afljdasfl;j

test
dsfjal
buf is dsfjal
fl;j

test

```

## nonblocking1.c

```
$ gcc nonblocking1.c -o nonblocking1
$ ./nonblocking1
read /dev/tty: Resource temporarily unavailable
no input,buf is null
read /dev/tty: Resource temporarily unavailable
no input,buf is null
read /dev/tty: Resource temporarily unavailable
no input,buf is null
read /dev/tty: Resource temporarily unavailable
no input,buf is null
dhread /dev/tty: Resource temporarily unavailable
no input,buf is null
elread /dev/tty: Resource temporarily unavailable
no input,buf is null
adjksf;a
ret = 13, buf is dheladjksf;a

adfsread /dev/tty: Resource temporarily unavailable
no input,buf is null
j;a
aret = 8, buf is adfsj;a
sf;a

dlsfj;
ret = 8, buf is adlsfj;
sf;a

^C
mac020:A1-blocking mac020$ 
```

## nonblocking2.c

```
mac020:A1-blocking mac020$ gcc nonblocking2.c -o nonblocking2
mac020:A1-blocking mac020$ ./nonblocking2
read /dev/tty: Resource temporarily unavailable
no input,buf is null
read /dev/tty: Resource temporarily unavailable
no input,buf is null
dadafljkread /dev/tty: Resource temporarily unavailable
no input,buf is null

adfjal;sret = 9, buf is dadafljk


adfs;dfhjaldret = 9, buf is adfjal;s


adlsfjal;sret = 13, buf is adfs;dfhjald


ret = 11, buf is adlsfjal;s
d

^C
```