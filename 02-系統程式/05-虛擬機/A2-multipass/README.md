# Ubuntu -- multipass

multipass 是從 docker 改過來專為 ubuntu 打造的容器版本。

* 安裝 -- https://ubuntu.com/download/server
    * 選 Get Multipass
* [介绍一款 Ubuntu 虚拟机管理神器 - Multipass](https://segmentfault.com/a/1190000039416537)
* [Multipass虛擬機使用教程](https://codingnote.cc/zh-tw/p/287792/)

## Mac

Mac 安裝好後右上角有個橘色圓形圖示，選 Open Shell 就可以開啟一個新的 Linux。

## 用指令自建虛擬機

```
$ multipass launch -n ubuntu-lts -c 4 -m 4G -d 40G
```

## 用指令進入虛擬機

```
$ multipass shell ubuntu-lts
```

## 列出虛擬機

## mac 問題

* https://github.com/canonical/multipass/issues/1447

若建立失敗，應關掉 internet sharing.
