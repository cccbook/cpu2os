
# tmux

* [阮一峰: Tmux 使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html)

## 最简操作流程

1. 新建会话tmux new -s my_session。
2. 在 Tmux 窗口运行所需的程序。
3. 按下快捷键Ctrl+b d将会话分离。
4. 下次使用时，重新连接到会话tmux attach-session -t my_session。

## 快捷鍵

```
Ctrl-B % for a vertical split (one shell on the left, one shell on the right)
Ctrl-B" for a horizontal split (one shell at the top, one shell at the bottom)
Ctrl-B O to make the other shell active
Ctrl-B ? for help
Ctrl-B d detach from Tmux, leaving it running in the background (use tmux attach to reenter)
```
