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
