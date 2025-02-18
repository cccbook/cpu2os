
* https://blog.gtwang.org/linux/screen-command-examples-to-manage-linux-terminals/2/

1. 打 screen 進入

    在 screen 工作環境中若要建立新的視窗，可以按下 Ctrl + a 後，再按下 c 鍵（create），這樣就會建立一個新的視窗，並出現一個獨立的互動式 shell。

    若要在不同的視窗之間切換，可以按下 Ctrl + a 後，再按下 n 鍵（next），這樣就會切換至下一個視窗，若要切換至錢一個視窗，則可以按下 Ctrl + a 後，再按下 p 鍵（previous）。

2. 打 Ctrl-A-D 卸離

3. 打 screen -ls 列出連線

    screen -ls
    There is a screen on:
            11894.pts-0.localhost   (04/26/2023 02:34:01 AM)        (Detached)
    1 Socket in /run/screen/S-guest2.

4. 打 screen -r 連線名稱回去

    guest2@localhost:~/cpu2os$ screen -r 11894.pts-0.localhost


5. 真正離開

    若要離開 screen 工作環境有幾種不同的方式，如果你的工作還沒做完，想要暫時離開，則使用上面介紹的卸離功能（Ctrl + a 與 d），而如果是工作都已經做完了，則可直接以 exit 指令離開 shell，在 screen 工作環境中所有的 shell 都結束後，screen 會自動結束，另外也可以使用 Ctrl + a 與 k（kill），直接關閉整個 screen 工作環境。