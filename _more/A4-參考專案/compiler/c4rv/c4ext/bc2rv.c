#include "c4.h"
// #include "../rv/riscv.h"

/*
1. 由於指令代碼可能來自 ct 與 rv，兩者都有可能有 ADD, SUB，但由大小小仍可以區分：

ct: ADD, SUB, MUL, DIV, MOD, ....
rv: Add, Sub, Mul, Div, Rem, ....

2. printf 之類的指令，使用 ecall 的方式實踐。
_start: addi  a0, x0, 1      # 1 = StdOut
        la    a1, helloworld # load address of helloworld
        addi  a2, x0, 13     # length of our string
        addi  a7, x0, 64     # linux write system call
        ecall                # Call linux to output the string

enum { LVAR , ... OPEN,READ,WRIT,CLOS,PRTF,SPRT,SSCN,GETS,MALC,FREE,MSET,MCMP,MCPY,EXIT }
*/

/*
enum {Reg,Imm};
enum { Zero, Ra, Sp, Gp, Tp, T0, T1, T2,
   S0, Fp=8, S1, A0, A1, A2, A3, A4, A5,
       A6,   A7, S2, S3, S4, S5, S6, S7, 
       S8,   S9, S10,S11,T3, T4, T5, T6 };
*/
#define rx1 "t0"
#define rx2 "t1"
#define cmt(...) printf("# ");printf(__VA_ARGS__);printf("\n")
#define debug(...) fprintf(stderr, __VA_ARGS__)
#define label(...) printf(__VA_ARGS__);printf(":")
#define emit(...) printf("  ");printf(__VA_ARGS__);printf("\n")
// 偽指令
#define mov(rd, rs) emit("add %s,%s,x0", rd, rs)
#define li(rd, imm) emit("addi %s,x0,%d", rd, imm)
#define not(rd, rs) emit("xori %s,%s,-1", rd, rs)
#define seqz(rd, rs) emit("sltiu %s,%s,1", rd, rs)
#define snez(rd, rs) emit("sltu %s,x0,%s", rd, rs)
#define slt(rd, rs1, rs2) emit("slt %s,%s,%s", rd, rs1, rs2)
#define sgt(rd, rs1, rs2) emit("slt %s,%s,%s", rd, rs2, rs1)
#define seq(rd, rs1, rs2) emit("sub %s,%s,%s", rx1, rs1, rs2);seqz(rd, rx1)
#define sne(rd, rs1, rs2) emit("sub %s,%s,%s", rx1, rs1, rs2);snez(rd, rx1)
#define sge(rd, rs1, rs2) sgt(rx1, rs1, rs2);seq(rx2,rs1,rs2);emit("or %s,%s,%s", rd, rx1, rx2)
#define sle(rd, rs1, rs2) slt(rx1, rs1, rs2);seq(rx2,rs1,rs2);emit("or %s,%s,%s", rd, rx1, rx2)
#define beq(rs1,rs2,offset) emit("beq %s,%s,%d", rs1, rs2, offset)
#define bne(rs1,rs2,offset) emit("bne %s,%s,%d", rs1, rs2, offset)
#define blt(rs1,rs2,offset) emit("blt %s,%s,%d", rs1, rs2, offset)
#define bge(rs1,rs2,offset) emit("bge %s,%s,%d", rs1, rs2, offset)
#define bgt(rs1,rs2,offset) emit("blt %s,%s,%d", rs2, rs1, offset)
#define ble(rs1,rs2,offset) emit("bge %s,%s,%d", rs2, rs1, offset)
#define ret() emit("jalr x0, x1, 0")

char *r = "s0";
char *a = "s1";
char *rimm = "s2";
char *bp = "s3";
char *rstab = "s4";
char *rdata = "s5";

void push(char *reg) {      // *--sp = reg;
  emit("addi sp,sp,-1");    // --sp
  emit("sw %s,0(sp)", reg); // *sp=reg
}

char* pop() { // r = *sp++
  emit("lw %s,0(sp)", r); // r=*sp;
  emit("addi sp,sp,1"); // *sp++;
  return r;
}

void op(char *op0) { // a = *sp++ op0 a;
  char *t=pop();
  emit("%s %s,sp,%s", op0, a, t);
}

char timm[100];
char *imm(int value) { sprintf(timm, "%d", value); return timm; }

#define PC_ADD_SIZE 2 // 這需要根據 pc_add 的指令數來修改。
void pc_add(int pc0, int offset) {
  emit("beq x0,x0,pc%d # pc_add(%d,%d)", pc0+offset, pc0, offset);
}

void scall(int id) {
  li("a7", id);
  emit("ecall");
}

/* 注意，光 bytecode 裡的碼是不夠的，還得搭配前置的 stack 代碼才行。
int vm_main(int *pc, int argc, char **argv) {
  int *bp, *sp, *t;
  // setup stack
  bp = sp = (int *)((int)stack + poolsz); // 將 sp, bp 指向堆疊起始點 (底端)
  *--sp = EXIT; // call exit if main returns，執行到此 EXIT 指令就會離開 vm_run()
  *--sp = PSH;  // ??
  t = sp;       // ??
  *--sp = argc; // argc, argv 是要傳給 main 的參數。
  *--sp = (int)argv;
  *--sp = (int)t;  // ??
  vm_run(pc, bp, sp, data, stab); // 開始執行 pc 指向的 main(argc,argv) 函數
  return 0;
}
*/
void bc_to_rv() {
  int codeLen = h[CSize]/W;
  int *pc = code;
  char *t;
  for (int i=0; pc-code<codeLen; i++) {
    label("pc%d", pc-code);
    int ir = *pc++;
    int pc0 = pc-code;
    // printf("%04X:%08X ", i, ir);
    switch (ir) {
      case LVAR: cmt("LVAR %d", *pc); // a = (int)(bp + *pc++);
        emit("addi %s,%s,%s", a, bp, imm(*pc++));
        break;
      case IMM: cmt("IMM %d", *pc); // a = *pc++;
        emit("addi %s,x0,%s", a, imm(*pc++));
        break;
      case STR: cmt("STR %d", *pc); // a = (int)(stab+*pc++);
        emit("addi %s,%s,%s", a, rstab, imm(*pc++));
        break;
      case GVAR: cmt("GVAR %d", *pc); // a = (int)(data+*pc++);
        emit("addi %s,%s,%s", a, rdata, imm(*pc++));
        break;
      case JMP: cmt("JMP %d", *pc); // pc = (int *) (pc+*pc);
        pc_add(pc0,*pc); pc++;
        break; 
      case JSR: cmt("JSR %d", *pc); // { *--sp = (int)(pc + 1); pc = (int *)(pc+*pc); }
        push("sp");
        li("sp", (int)(pc-code+1)); // 這樣對嗎?? pc-code+1
        pc_add(pc0,*pc); pc++;
        break;
      case ENT: cmt("ENT %d", *pc); // { *--sp = (int)bp; bp = sp; sp = sp - *pc++; }
        push("bp");
        mov("bp", "sp");
        emit("addi sp, sp, %s", imm(-*pc)); pc++;
        break;
      case ADJ: cmt("ADJ %d", *pc); // sp = sp + *pc++;
        emit("addi sp, sp, %s", imm(*pc++));
        break;
      case BZ: cmt("BZ"); // pc = a ? pc + 1 : (int *) (pc+*pc);
        emit("beq %s,x0,%d", a, PC_ADD_SIZE);
        pc_add(pc0, *pc);
        break;
      case BNZ: cmt("BNZ"); // pc = a ? (int *)(pc+*pc) : pc + 1;
        emit("bne %s,x0,%d", a, PC_ADD_SIZE);
        pc_add(pc0, *pc);
        break;
      case LEV: cmt("LEV"); // { sp = bp; bp = (int *)*sp++; pc = (int *)*sp++; }
        mov("sp", "bp");
        t=pop("sp");
        mov("bp", t);
        break;
      case LI: cmt("LI"); // load int: a = *(int *)a;  
        emit("lw %s,0(%s)", a, a);
        break;
      case LC: cmt("LC"); // load char: a = *(char *)a;
        emit("lb %s,0(%s)", a, a);
        break;
      case SI: cmt("SI"); // store int *(int *)*sp++ = a;
        t=pop();
        emit("sw %s,0(%s)", a, t);
        break;
      case SC: cmt("SC"); // store char a = *(char *)*sp++ = a;
        t=pop();
        emit("sb %s,0(%s)", a, t);
        break;
      case PSH: cmt("PSH"); // *--sp = a;
        push(a);
        break;
      // 運算指令
      case OR: cmt("OR"); // a = *sp++ |  a;
        op("or");
        break;
      case XOR: cmt("XOR"); // a = *sp++ ^  a;
        op("xor");
        break;
      case AND: cmt("AND"); // a = *sp++ &  a;
        op("and");
        break;
      case EQ: cmt("EQ"); // a = *sp++ == a;
        t = pop();
        seq(a, a, t);
        break;
      case NE: cmt("NE"); // a = *sp++ != a;
        t = pop();
        sne(a, a, t);
        break;
      case LT: cmt("LT"); // a = *sp++ <  a;
        t = pop();
        slt(a, a, t);
        break;
      case GT: cmt("GT"); // a = *sp++ >  a;
        t = pop();
        sgt(a, a, t);
        break;
      case LE: cmt("LE"); // a = *sp++ <= a;
        t = pop();
        sle(a, a, t);
        break;
      case GE: cmt("GE"); // a = *sp++ >= a;
        t = pop();
        sge(a, a, t);
        break;
      case SHL: cmt("SHL"); // a = *sp++ << a;
        op("sll");
        break;
      case SHR: cmt("SHR"); // a = *sp++ >> a;
        op("srl");
        break;
      case ADD: cmt("ADD"); // a = *sp++ +  a;
        op("add");
        break;
      case SUB: cmt("SUB"); // a = *sp++ -  a;
        op("sub");
        break;
      case MUL: cmt("MUL"); // a = *sp++ *  a;
        op("mul");
        break;
      case DIV: cmt("DIV"); // a = *sp++ /  a;
        op("div");
        break;
      case MOD: cmt("MOD"); // a = *sp++ %  a;
        op("rem");
        break;
      // 系統呼叫
      case OPEN: cmt("OPEN"); // a = open((char *)sp[2], sp[1], *sp);
        scall(OPEN);
        break;
      case READ: cmt("READ"); // a = read(sp[2], (char *)sp[1], *sp);
        scall(READ);
        break;
      case WRIT: cmt("WRIT"); // a = write(sp[2], (char *)sp[1], *sp);
        scall(WRIT);
        break;
      case CLOS: cmt("CLOS"); // close(*sp);
        scall(CLOS);
        break;
      case PRTF: cmt("PRTF"); // { t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]); }
        scall(PRTF);
        break;
      case SPRT: cmt("SPRT"); // { t = sp + pc[1]; a = sprintf((char *)t[-1], (char *)t[-2], t[-3], t[-4], t[-5], t[-6]); } // sprintf(str,"....", a, b, c, d, e)
        scall(SPRT);
        break;
      case SSCN: cmt("SSCN"); // { t = sp + pc[1]; a = sscanf((char *)t[-1], (char *)t[-2], t[-3], t[-4], t[-5], t[-6]); } // sprintf(str,"....", a, b, c, d, e)
        scall(SSCN);
        break;
      case GETS: cmt("GETS"); // gets((char*)*sp);
        scall(GETS);
        break;
      case MALC: cmt("MALC"); // malloc
        scall(MALC);
        break;
      case FREE: cmt("FREE"); // free((void *)*sp);
        scall(FREE);
        break;
      case MSET: cmt("MSET"); // a = (int)memset((char *)sp[2], sp[1], *sp);
        scall(MSET);
        break;
      case MCMP: cmt("MCMP"); // a = memcmp((char *)sp[2], (char *)sp[1], *sp);
        scall(MCMP);
        break; 
      case MCPY: cmt("MCPY"); // a = (int)memcpy((char *)sp[2], (char *)sp[1], *sp);
        scall(MCPY);
        break;
      case EXIT: cmt("EXIT"); // return
        scall(EXIT);
        break;
      default: debug("Error: unknown command %d\n", ir);
    }
    printf("\n");
  }
  // dump_imm();
}

int main(int argc, char **argv) { // 主程式
  oFile = argv[1];
  init();
  bc_load();
  bc_to_rv();
}

/*
int immData[9999];
int immIdx = 0;

char* immRange(int value, int from, int to) {
  if (imm >= from && imm <=to) {
    sprintf(timm, "%d", value);
    return timm;
  } else {
    emit("lw %s,%d(%s)", r, immIdx*4, rimm);
    immData[immIdx++] = value;
    return r;
  }
}

char *imm12(int value) { return immRange(value, -2048, 2047); }
char *immB(int value) { return immRange(value, -4096, 4095); }
char *immU(int value) { return immRange(value, -2147483648, 2147483647); }
char *immJ(int value) { return immRange(value, -1048576, 1048575); }
void dump_imm() {
  printf("imm:");
  for (int i=0; i<immIdx; i++) {
    printf("%d,", immData[i]);
  }
  printf("\n");
}

*/
