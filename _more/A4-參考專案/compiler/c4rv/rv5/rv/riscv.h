#ifndef __RISCV_H__
#define __RISCV_H__

#include "../lib/lib.h"
#include "../lib/map.h"
#include "../lib/st.h"

#define MMAX 32768

enum Id { 
  None, 
  /*RV32I 運算指令*/ 
  Sll, 
  Slli, 
  Srl, 
  Srli, 
  Sra, 
  Srai, 
  Add, 
  Addi, 
  Sub, 
  Xor, 
  Xori, 
  Or, 
  Ori, 
  And, 
  Andi,
  // 比較設定
  Slt, 
  Slti, 
  Sltu, 
  Sltiu, 
  // 跳躍指令
  Beq, 
  Bne, 
  Blt, 
  Bge, 
  Bltu, 
  Bgeu, 
  Jal, 
  Jalr, 
  // 載入儲存
  Lb, 
  Lh, 
  Lw, 
  Lbu, 
  Lhu, 
  Sb, 
  Sh, 
  Sw,
  // 控制暫存器指令
  Csrrw, 
  Csrrs, 
  Csrrc, 
  Csrrwi, 
  Csrrsi, 
  Csrrci, 
  // 特殊指令
  Lui, 
  Auipc, 
  Fence, 
  Fencei, 
  Ecall, 
  Ebreak,
  /*RV32M 乘除法指令 */
  Mul, 
  Mulh, 
  Mulhsu, 
  Mulhu, 
  Div, 
  Divu, 
  Rem, 
  Remu
};

enum { Zero, Ra, Sp, Gp, Tp, T0, T1, T2,
   S0, Fp=8, S1, A0, A1, A2, A3, A4, A5,
       A6,   A7, S2, S3, S4, S5, S6, S7, 
       S8,   S9, S10,S11,T3, T4, T5, T6 };

typedef struct _Op {
  enum Id id;
  char *name;
  char type;
  uint32_t op, f3, f7;
} Op;

typedef struct _Ki {
  char *k;
  int  i;
} Ki;

extern Op opList[];
extern Map opMap;
extern Map rMap;

extern void riscvInit();
extern int rFind(char *name);

#endif

/*
// Machine-level CSRs.
#define MHARTID 0xf14 // Hardware thread ID.
#define MSTATUS 0x300 // Machine status register.
#define MEDELEG 0x302 // Machine exception delefation register.
#define MIDELEG 0x303 // Machine interrupt delefation register.
#define MIE 0x304 // Machine interrupt-enable register.
#define MTVEC 0x305 // Machine trap-handler base address.
#define MCOUNTEREN 0x306 // Machine counter enable.
#define MSCRATCH 0x340 // Scratch register for machine trap handlers.
#define MEPC 0x341 // Machine exception program counter.
#define MCAUSE 0x342 // Machine trap cause.
#define MTVAL 0x343 // Machine bad address or instruction.
#define MIP 0x344 // Machine interrupt pending.

// Supervisor-level CSRs.
#define SSTATUS 0x100 // Supervisor status register.
#define SIE 0x104 // Supervisor interrupt-enable register.
#define STVEC 0x105 // Supervisor trap handler base address.
#define SSCRATCH 0x140 // Scratch register for supervisor trap handlers.
#define SEPC 0x141 // Supervisor exception program counter.
#define SCAUSE 0x142 // Supervisor trap cause.
#define STVAL 0x143 // Supervisor bad address or instruction.
#define SIP 0x144 // Supervisor interrupt pending.
#define SATP 0x180 // Supervisor address translation and protection.

*/
/* 指令型態
  | R, I, S, B, U, J
--|--------------------------------------------
R | f7 rs2 rs1 f3 rd op
I |   i12  rs1 f3 rd op
S | i7 rs2 rs1 f3 i5 op  i7=i[11:5] i5=i[4:0]
B | i7 rs2 rs1 f3 i5 op  i7=i[12|10:5] i5=i[4:1|11]
U |   i20         rd op  i20=i[31:12]
J |   i20         rd op  i20=i[20|10:1|11|19:12]
*/