        org 07c00h
        mov ax, cs
        mov ds, ax
        mov es, ax
        call DispStr
        jmp $
    DispStr:
        mov ax, BM
        mov bp, ax
        mov cx, 16
        mov ax, 01301h
        mov bx, 000ch
        mov dl, 0
        int 10h
        ret
    BM:  db "Hello world!"
    times 510-($-$$) db 0
 dw 0xaa55

作者：James Liu
链接：https://www.zhihu.com/question/38424683/answer/1242910077
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。