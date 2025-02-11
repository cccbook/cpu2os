# 容器 -- 類似 docker

* https://blog.lizzie.io/linux-containers-in-500-loc.html
    * https://blog.lizzie.io/linux-containers-in-500-loc/contained.c
    * https://github.com/osinstom/containers-impl-c

## 安裝與編譯

```
sudo apt-get install libseccomp-dev libcap-dev
ls /usr/include/sys/capability.h
gcc -Wall -Werror contained.c -o contained -lcap -lseccomp
```

## 執行

我的執行

```
root@localhost:~/cpu2os/A4-參考專案/vm/contained# ./contained -m ~/misc/busybox-im
g/ -u 0 -c /bin/sh
=> validating Linux version...expected 4.7.x or 4.8.x: 6.8.0-41-generic
```

但應該要有如下結果

```
[lizzie@empress l-c-i-500-l]$ sudo ./contained -m ~/misc/busybox-img/ -u 0 -c /bin/sh
=> validating Linux version...4.7.10.201610222037-1-grsec on x86_64.
=> setting cgroups...memory...cpu...pids...blkio...done.
=> setting rlimit...done.
=> remounting everything with MS_PRIVATE...remounted.
=> making a temp directory and a bind mount there...done.
=> pivoting root...done.
=> unmounting /oldroot.oQ5jOY...done.
=> trying a user namespace...writing /proc/32627/uid_map...writing /proc/32627/gid_map...done.
=> switching to uid 0 / gid 0...done.
=> dropping capabilities...bounding...inheritable...done.
=> filtering syscalls...done.
/ # whoami
root
/ # hostname
05fe5c-three-of-pentacles
/ # exit
=> cleaning cgroups...done.
```

所以看來失敗了！

## ccc: 我執行 contained 的時候，發現 ... 失敗了！ 可能是什麼原因呢？

根據你提供的錯誤信息和預期的執行結果，問題出在程式在驗證 Linux 版本時失敗了。以下是可能的原因和解決方案：

### 可能的原因

1. **內核版本不匹配**：
   - 錯誤信息顯示你的 Linux 版本是 `6.8.0-41-generic`，而預期的版本範圍是 `4.7.x` 或 `4.8.x`。這意味著 `contained` 程式的設計可能是針對較舊的內核版本進行開發的。

2. **容器功能支援不足**：
   - 你正在運行的內核版本可能不支持某些容器相關的功能，或者這些功能在新版本中發生了變化。

3. **未使用 root 權限執行**：
   - 雖然你的命令中使用了 `root` 用戶，但如果程式內部仍然依賴 `sudo` 或其他方式來確保根權限，可能會影響結果。確保在執行時具有足夠的權限。

### 解決方案

1. **檢查內核版本**：
   - 如果你的環境允許，考慮使用支持 `contained` 所需的內核版本（例如 4.7.x 或 4.8.x）。你可以使用虛擬機器或 Docker 容器運行較舊的內核版本進行測試。

2. **修改驗證邏輯**：
   - 如果你有權限修改 `contained` 程式碼，考慮更新版本驗證邏輯以接受當前的內核版本。注意這可能導致不兼容的問題，因此需要謹慎處理。

3. **使用舊版本的程式**：
   - 如果可能，尋找或使用與你當前內核版本相容的舊版本 `contained` 程式碼。

4. **檢查內核配置**：
   - 確保你的內核已啟用所有必需的容器和 namespace 支持功能。可以查看 `/boot/config-$(uname -r)` 文件中的設置。

5. **參考文檔或社群**：
   - 查看 `contained` 的文檔或相關的社群（如 GitHub issues、論壇等），以了解是否有其他用戶遇到類似問題並找到解決方案。

透過這些方法，你應該能夠找到問題的根本原因，並可能解決 `contained` 的執行失敗問題。