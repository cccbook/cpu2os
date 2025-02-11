# 賈伯斯、IBM PC 的 BIOS 與台灣的電腦產業

1976年4月，[史蒂夫·賈伯斯](https://zh.wikipedia.org/wiki/%E5%8F%B2%E8%92%82%E5%A4%AB%C2%B7%E4%B9%94%E5%B8%83%E6%96%AF)、[斯蒂夫·沃茲尼亞克](https://zh.wikipedia.org/wiki/%E6%96%AF%E8%92%82%E5%A4%AB%C2%B7%E6%B2%83%E5%85%B9%E5%B0%BC%E4%BA%9A%E5%85%8B)和[羅納德·韋恩](https://zh.wikipedia.org/wiki/%E7%BD%97%E7%BA%B3%E5%BE%B7%C2%B7%E9%9F%A6%E6%81%A9)創立了蘋果公司，第一款推出的個人電腦是 [Apple I](https://zh.wikipedia.org/wiki/Apple_I) ，這款電腦雖然銷售只有幾台，但卻引出後來大賣的 [Apple II](https://en.wikipedia.org/wiki/Apple_II) 。

![](https://zh.wikipedia.org/wiki/File:Apple_I_Computer.jpg)

Apple II 推出之後，《個人電腦》這個在大型電腦龍頭 IBM 眼中只是玩具的裝置，卻成為席捲全球的重要科技產品。 Apple 也因此成為重要的科技公司。

![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Apple_II_typical_configuration_1977.png/330px-Apple_II_typical_configuration_1977.png)

IBM 覺得《個人電腦》只是個玩具，並非重要的產品，但卻又不想看著 Apple 長大成為 IBM 的對手，於是設計了一組《個人電腦架構》，並在 1981 年釋出。

這個在唐·埃斯特利奇領導下的12人小組用了約一年的時間研製出了IBM PC。為了達到這個目的他們首先決定使用現成的、不同[原始裝置製造商](https://zh.wikipedia.org/wiki/OEM)的元件。這個做法與IBM過去始終研製自己的元件的做法相反。其次他們決定使用[開放結構](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%B3%BB%E7%BB%9F%E7%BB%93%E6%9E%84)，這樣其它生產商可以生產和出售相容的元件和軟體。

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/IBM_PC-IMG_7271_%28transparent%29.png/375px-IBM_PC-IMG_7271_%28transparent%29.png)

IBM 還出售其《IBM PC技術參考資料》，這份資料中包括一段[ROM](https://zh.wikipedia.org/wiki/%E5%8F%AA%E8%AF%BB%E5%AD%98%E5%82%A8%E5%99%A8)[BIOS](https://zh.wikipedia.org/wiki/BIOS)[原始碼](https://zh.wikipedia.org/wiki/%E6%BA%90%E4%BB%A3%E7%A0%81)。[[1]](https://zh.wikipedia.org/zh-tw/IBM_PC#cite_note-1)[[2]](https://zh.wikipedia.org/zh-tw/IBM_PC#cite_note-2)。

其它生產商很快就[逆向分析](https://zh.wikipedia.org/wiki/%E9%80%86%E5%90%91%E5%B7%A5%E7%A8%8B)了BIOS的程式，發展了其自己的、不侵犯著作權的拷貝。1982年6月[哥倫比亞資料產品公司](https://zh.wikipedia.org/w/index.php?title=%E5%93%A5%E4%BC%A6%E6%AF%94%E4%BA%9A%E6%95%B0%E6%8D%AE%E4%BA%A7%E5%93%81%E5%85%AC%E5%8F%B8&action=edit&redlink=1)（Columbia Data Products）推出了第一台IBM PC相容機。1982年11月[康柏電腦](https://zh.wikipedia.org/wiki/%E5%BA%B7%E6%9F%8F%E9%9B%BB%E8%85%A6)宣布發展出第一台IBM PC相容的可攜式電腦[Compaq Portable](https://zh.wikipedia.org/w/index.php?title=Compaq_Portable&action=edit&redlink=1)（1983年3月出產）。

台灣的廠商，像是宏碁電腦，1981年就在研發自己的單板電腦 [小教授一號](https://zh.m.wikipedia.org/zh-tw/%E5%B0%8F%E6%95%99%E6%8E%88%E4%B8%80%E5%8F%B7) ，之後還推出了 [模仿 Apple II 的小教授二號機型](https://zh.m.wikipedia.org/zh-tw/%E5%B0%8F%E6%95%99%E6%8E%88%E4%BA%8C%E5%8F%B7)。

宏碁電腦的創辦人施振榮，在一九八二年的拉斯維加斯的 Comdex Show 裡，看到康柏一個沒沒無名的公司，因為做一個 IBM PC 相容電腦而造成轟動，於是回國後就宏碁就開始做 IBM PC，在一九八四年年底就推出了產品。

那時候其實還有其他公司，像是 迪吉多、王安等都推出個人電腦，但是和 IBM PC 不相容，因此在 IBM PC 市場做大之後，迪吉多、王安反而都愈來愈差，終於被市場淘汰了！

IBM PC 規格中的 [BIOS](https://zh.wikipedia.org/zh-tw/BIOS) 規格被公開，然後又被逆向工程解出來，這件事情非常關鍵，因為 BIOS 是軟硬體兩者的介面，具有連接軟硬體兩者的功能。

有了 BIOS，就能將軟硬體兩個工業切開，軟體廠商專心做軟體，硬體廠商專心做硬體，只要能夠和 BIOS 接上就行了。

IBM PC 的 BIOS 掌控在 [Phoenix 鳳凰科技](https://www.104.com.tw/company/dbaijuo) 手上，然後軟體作業系統被微軟買下的 DOS 搶到，而硬體則被美商《康柏》與台灣的宏碁等公司等搶到，這些公司都使用 Intel 的 80x86 系列處理器，於是形成了 Microsoft+Intel+Taiwan 的 MIT 三足鼎立產業鏈。

## BIOS 與 DOS

BIOS 到底是甚麼？可以將軟硬體切開得如此徹底呢？

我大學時期親身體驗過這個技術，很清楚 BIOS 的功效。

只要看一下《施威銘》寫的 [DOS技術手冊(一)(二)(三)](http://album.udn.com/tchcvsdp/photo/2482162?f_number=2) ，其實就能理解箇中奧妙了。

![](https://g.udn.com.tw/community/img/PSN_PHOTO/tchcvsdp/f_2482162_1.JPG)

以下是一段可以在 IBM PC 下印出 "Hello!" 的組合語言，其中的 int 21h 是 DOS 的系統呼叫，其實就是呼叫 BIOS 提供的系統函數。

```
code segment                    ; start    段開始位址
assume cs:code,ds:code          ; 設定程式段及資料段
org 100h                        ; 起始位址

start: jmp begin                ; 程式進入點
    msg db 'Hello!$'            ; 要印出的訊息
begin: mov dx,offset msg        ; 設定參數 ds:dx = 字串起點
    mov ah,9                    ; 設定9號服務
    int 21h                     ; 進行DOS系統呼叫
    mov ax,4c00h                ; 設定4C號服務
    int 21h                     ; 進行DOS系統呼叫
code    ends                    ; .code 段結束
end                             ; 程式結束點
```

於是螢幕製造商就不需要和微軟溝通，只要能支援 BIOS 規格就行了。

接著看看 INT 13H 讀寫磁碟功能的程式範例：

```
   [ORG 7c00h]   ; code starts at 7c00h
   xor ax, ax    ; make sure ds is set to 0
   mov ds, ax
   cld
   ; start putting in values:
   mov ah, 2h    ; int13h function 2
   mov al, 63    ; we want to read 63 sectors
   mov ch, 0     ; from cylinder number 0
   mov cl, 2     ; the sector number 2 - second sector (starts from 1, not 0)
   mov dh, 0     ; head number 0
   xor bx, bx    
   mov es, bx    ; es should be 0
   mov bx, 7e00h ; 512bytes from origin address 7c00h
   int 13h
   jmp 7e00h     ; jump to the next sector
   
   ; to fill this sector and make it bootable:
   times 510-($-$$) db 0 
   dw 0AA55h
```

有了 INT 13H ，硬碟製造商也不需要和微軟商量，就能自行生產與改進硬碟科技了。

BIOS 一刀切開了 PC 的《軟體、主機、螢幕、硬碟》等元件，於是各個產業就能獨立發展了！





