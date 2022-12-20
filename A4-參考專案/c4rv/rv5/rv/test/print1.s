  addi x1, x0, helloworld
  ecall
  beq x0, x0, .exit

helloworld: .ascii "Hello World!\n"
.exit: