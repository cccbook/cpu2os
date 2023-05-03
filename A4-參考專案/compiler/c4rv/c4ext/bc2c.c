#include "c4.h"

#define cmt(...) printf(" /*");printf(__VA_ARGS__);printf("*/ ")
#define label(i) printf("pc%d:", i)

void emitIntArray(char *name, int *array, int len) {
  emit("static int %s[]={", name);
  for (int i=0; i<len; i++) {
    emit("%d", array[i]);
    if (i<len-1) emit(",");
  }
  emit("};\n")
}

void emitCharArray(char *name, char *array, int len) {
  emit("static char %s[]={", name);
  for (int i=0; i<len; i++) {
    emit("%d", array[i]);
    if (i<len-1) emit(",");
  }
  emit("};\n")
}

void emitPcLabels(int len) {
  emit("static const void* _pc[]={");
  for (int i=0; i<len; i++) {
    emit("&&pc%d", i);
    if (i<len-1) emit(",");
  }
  emit("};\n")
}

char* go2pc() {
  static char tempstr[100];
  sprintf(tempstr, "goto *_pc[pc-code];"); // goto *(&&pc0 + pc[i]);
  return tempstr;
}

void bc2head() {
  emit("#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <string.h>\n");
  emit("int run() {\n\n");
  emitIntArray("code", code, codeLen);
  emitCharArray("stab", stab, stabLen);
  emitCharArray("data", data, dataLen);
  emit("static int stack[%d];\n", STACK_SIZE);
  emitPcLabels(codeLen);
  emit("int *pc=code+%d, *sp = stack+%d, *bp=sp, a, *t;\n", h[Entry], STACK_SIZE);
  emit("%s\n", go2pc());// pc 一開始並不一定是 0，因為 main() 沒宣告在最前面，所以要先跳到 pc。
}

void bc2fetch() { label(pc-code); emit("pc++;"); }

void bc2tail() {
  emit("}\nint main() {\n  run();\n}\n");
}

void bc2imm() { cmt("IMM %d", *pc); label(pc-code); pc++; emit("a = *pc++;"); }
void bc2str() { cmt("STR %d", *pc); label(pc-code); pc++; emit("a = (int)(stab+*pc++);"); }
void bc2lvar() { cmt("LVAR %d", *pc); label(pc-code); pc++; emit("a = (int)(bp + *pc++);"); }
void bc2gvar() { cmt("GVAR %d", *pc); label(pc-code); pc++; emit("a = (int)(data+*pc++);"); }
void bc2jmp() { cmt("JMP %d", *pc); label(pc-code); pc++; emit("pc = (int *) (pc+*pc); %s", go2pc()); }
void bc2jsr() { cmt("JSR %d", *pc); label(pc-code); pc++; emit("*--sp = (int)(pc + 1); pc = (int *)(pc+*pc); %s", go2pc()); }
void bc2ent() { cmt("ENT %d", *pc); label(pc-code); pc++; emit("*--sp = (int)bp; bp = sp; sp = sp - *pc++;"); }
void bc2adj() { cmt("ADJ %d", *pc); label(pc-code); pc++; emit("sp = sp + *pc++;"); }
void bc2bz() { cmt("BZ %d", *pc); label(pc-code); pc++; emit("if (!a) { pc=(int *)(pc+*pc); %s} else pc++;", go2pc()); }
void bc2bnz() { cmt("BNZ %d", *pc); label(pc-code); pc++; emit("if (a) { pc=(int *)(pc+*pc); %s} else pc++;", go2pc());  }
void bc2lev() {
  cmt("LEV"); // { sp = bp; bp = (int *)*sp++; pc = (int *)*sp++; }
  emit("sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;");
  emit("if (pc==NULL) return 0; else %s", go2pc()); // 當 LEV 到最上層時，直接離開。}
}
void bc2li() { cmt("LI"); emit("a = *(int *)a;"); }
void bc2lc() { cmt("LC"); emit("a = *(char *)a;"); }
void bc2si() { cmt("SI"); emit("*(int *)*sp++ = a;"); }
void bc2sc() { cmt("SC"); emit("a = *(char *)*sp++ = a;"); }
void bc2psh() { cmt("PSH"); emit("*--sp = a;");  }
void bc2or() { cmt("OR"); emit("a = *sp++ |  a;"); }
void bc2xor() { cmt("XOR"); emit("a = *sp++ ^  a;"); }
void bc2and() { cmt("AND"); emit("a = *sp++ &  a;"); }
void bc2eq() { cmt("EQ"); emit("a = *sp++ == a;"); }
void bc2ne() { cmt("NE"); emit("a = *sp++ != a;"); }
void bc2lt() { cmt("LT"); emit("a = *sp++ <  a;"); }
void bc2gt() { cmt("GT"); emit("a = *sp++ >  a;"); }
void bc2le() { cmt("LE"); emit("a = *sp++ <= a;"); }
void bc2ge() { cmt("GE"); emit("a = *sp++ >= a;"); }
void bc2shl() { cmt("SHL"); emit("a = *sp++ << a;"); }
void bc2shr() { cmt("SHR"); emit("a = *sp++ >> a;"); }
void bc2add() { cmt("ADD"); emit("a = *sp++ +  a;"); }
void bc2sub() { cmt("SUB"); emit("a = *sp++ -  a;"); }
void bc2mul() { cmt("MUL"); emit("a = *sp++ *  a;"); }
void bc2div() { cmt("DIV"); emit("a = *sp++ /  a;"); }
void bc2mod() { cmt("MOD"); emit("a = *sp++ %%  a;"); }
void bc2open() { cmt("OPEN"); emit("a = open((char *)sp[2], sp[1], *sp);"); }
void bc2read() { cmt("READ"); emit("a = read(sp[2], (char *)sp[1], *sp);"); }
void bc2writ() { cmt("WRIT"); emit("a = write(sp[2], (char *)sp[1], *sp);"); }
void bc2clos() { cmt("CLOS"); emit("close(*sp);"); }
void bc2prtf() { cmt("PRTF"); emit("t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);"); }
void bc2sprt() { cmt("SPRT"); emit("t = sp + pc[1]; a = sprintf((char *)t[-1], (char *)t[-2], t[-3], t[-4], t[-5], t[-6]);"); }
void bc2sscn() { cmt("SSCN"); emit("t = sp + pc[1]; a = sscanf((char *)t[-1], (char *)t[-2], t[-3], t[-4], t[-5], t[-6]);"); }
void bc2gets() { cmt("GETS"); emit("gets((char*)*sp);"); }
void bc2malc() { cmt("MALC"); emit("a = (int)malloc(*sp);"); }
void bc2free() { cmt("FREE"); emit("free((void *)*sp);"); }
void bc2mset() { cmt("MSET"); emit("a = (int)memset((char *)sp[2], sp[1], *sp);"); }
void bc2mcmp() { cmt("MCMP"); emit("a = memcmp((char *)sp[2], (char *)sp[1], *sp);"); }
void bc2mcpy() { cmt("MCPY"); emit("a = (int)memcpy((char *)sp[2], (char *)sp[1], *sp);"); }
void bc2exit() { cmt("EXIT"); emit("return"); }

#include "bc2x.c"
