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