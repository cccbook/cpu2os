#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
int run() {

static int code[]={8,1,1,-1,15,0,0,13,3,0,15,1,-1,11,15,36,9,2,1,-1,11,15,0,10,21,6,26,3,8,15,1,-1,11,15,36,9,2,1,-1,15,11,15,0,1,27,13,15,0,1,28,4,-33,10,8,0,5,-56,10};
static char stab[]={105,61,37,100,10,0,0,0,104,101,108,108,111,33,32,105,61,37,100,10,0,0,0,0};
static char data[]={};
static int stack[10000];
static const int _pc[]={&&pc0-&&pc0,&&pc1-&&pc0,&&pc2-&&pc0,&&pc3-&&pc0,&&pc4-&&pc0,&&pc5-&&pc0,&&pc6-&&pc0,&&pc7-&&pc0,&&pc8-&&pc0,&&pc9-&&pc0,&&pc10-&&pc0,&&pc11-&&pc0,&&pc12-&&pc0,&&pc13-&&pc0,&&pc14-&&pc0,&&pc15-&&pc0,&&pc16-&&pc0,&&pc17-&&pc0,&&pc18-&&pc0,&&pc19-&&pc0,&&pc20-&&pc0,&&pc21-&&pc0,&&pc22-&&pc0,&&pc23-&&pc0,&&pc24-&&pc0,&&pc25-&&pc0,&&pc26-&&pc0,&&pc27-&&pc0,&&pc28-&&pc0,&&pc29-&&pc0,&&pc30-&&pc0,&&pc31-&&pc0,&&pc32-&&pc0,&&pc33-&&pc0,&&pc34-&&pc0,&&pc35-&&pc0,&&pc36-&&pc0,&&pc37-&&pc0,&&pc38-&&pc0,&&pc39-&&pc0,&&pc40-&&pc0,&&pc41-&&pc0,&&pc42-&&pc0,&&pc43-&&pc0,&&pc44-&&pc0,&&pc45-&&pc0,&&pc46-&&pc0,&&pc47-&&pc0,&&pc48-&&pc0,&&pc49-&&pc0,&&pc50-&&pc0,&&pc51-&&pc0,&&pc52-&&pc0,&&pc53-&&pc0,&&pc54-&&pc0,&&pc55-&&pc0,&&pc56-&&pc0,&&pc57-&&pc0};
int *pc=code+53, *sp = stack+10000, *bp=sp, a, *t;
goto *(&&pc0+_pc[pc-code]);
pc0:pc++; /*ENT 1*/ pc1:*--sp = (int)bp; bp = sp; sp = sp - *pc++;
pc2:pc++; /*LVAR -1*/ pc3:a = (int)(bp + *pc++);
pc4:pc++; /*PSH*/ *--sp = a;
pc5:pc++; /*IMM 0*/ pc6:a = *pc++;
pc7:pc++; /*SI*/ *(int *)*sp++ = a;
pc8:pc++; /*STR 0*/ pc9:a = (int)(stab+*pc++);
pc10:pc++; /*PSH*/ *--sp = a;
pc11:pc++; /*LVAR -1*/ pc12:a = (int)(bp + *pc++);
pc13:pc++; /*LI*/ a = *(int *)a;
pc14:pc++; /*PSH*/ *--sp = a;
pc15:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc16:pc++; /*ADJ 2*/ pc17:sp = sp + *pc++;
pc18:pc++; /*LVAR -1*/ pc19:a = (int)(bp + *pc++);
pc20:pc++; /*LI*/ a = *(int *)a;
pc21:pc++; /*PSH*/ *--sp = a;
pc22:pc++; /*IMM 10*/ pc23:a = *pc++;
pc24:pc++; /*LT*/ a = *sp++ <  a;
pc25:pc++; /*BZ 26*/ pc26:if (!a) { pc=(int *)(pc+*pc); goto *(&&pc0+_pc[pc-code]);} else pc++;
pc27:pc++; /*STR 8*/ pc28:a = (int)(stab+*pc++);
pc29:pc++; /*PSH*/ *--sp = a;
pc30:pc++; /*LVAR -1*/ pc31:a = (int)(bp + *pc++);
pc32:pc++; /*LI*/ a = *(int *)a;
pc33:pc++; /*PSH*/ *--sp = a;
pc34:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc35:pc++; /*ADJ 2*/ pc36:sp = sp + *pc++;
pc37:pc++; /*LVAR -1*/ pc38:a = (int)(bp + *pc++);
pc39:pc++; /*PSH*/ *--sp = a;
pc40:pc++; /*LI*/ a = *(int *)a;
pc41:pc++; /*PSH*/ *--sp = a;
pc42:pc++; /*IMM 1*/ pc43:a = *pc++;
pc44:pc++; /*ADD*/ a = *sp++ +  a;
pc45:pc++; /*SI*/ *(int *)*sp++ = a;
pc46:pc++; /*PSH*/ *--sp = a;
pc47:pc++; /*IMM 1*/ pc48:a = *pc++;
pc49:pc++; /*SUB*/ a = *sp++ -  a;
pc50:pc++; /*JMP -33*/ pc51:pc = (int *) (pc+*pc); goto *(&&pc0+_pc[pc-code]);
pc52:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *(&&pc0+_pc[pc-code]);
pc53:pc++; /*ENT 0*/ pc54:*--sp = (int)bp; bp = sp; sp = sp - *pc++;
pc55:pc++; /*JSR -56*/ pc56:*--sp = (int)(pc + 1); pc = (int *)(pc+*pc); goto *(&&pc0+_pc[pc-code]);
pc57:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *(&&pc0+_pc[pc-code]);
}
int main() {
  run();
}
