#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
int run() {

static int code[]={8,2,1,-1,15,0,0,13,1,-2,15,0,1,13,3,0,15,1,-2,11,15,1,-1,11,15,36,9,3,1,-2,11,15,1,2,11,23,6,42,3,12,15,1,-2,11,15,1,-1,11,15,36,9,3,1,-1,15,1,-1,11,15,1,-2,11,27,13,1,-2,15,11,15,0,1,27,13,15,0,1,28,4,-50,1,-1,11,10,10,8,0,3,24,15,0,10,15,5,-93,9,1,15,36,9,2,10};
static char stab[]={105,61,37,100,32,115,61,37,100,10,0,0,105,61,37,100,32,115,61,37,100,10,0,0,115,117,109,40,49,48,41,61,37,100,10,0};
static char data[]={};
static int stack[10000];
static const void* _pc[]={&&pc0,&&pc1,&&pc2,&&pc3,&&pc4,&&pc5,&&pc6,&&pc7,&&pc8,&&pc9,&&pc10,&&pc11,&&pc12,&&pc13,&&pc14,&&pc15,&&pc16,&&pc17,&&pc18,&&pc19,&&pc20,&&pc21,&&pc22,&&pc23,&&pc24,&&pc25,&&pc26,&&pc27,&&pc28,&&pc29,&&pc30,&&pc31,&&pc32,&&pc33,&&pc34,&&pc35,&&pc36,&&pc37,&&pc38,&&pc39,&&pc40,&&pc41,&&pc42,&&pc43,&&pc44,&&pc45,&&pc46,&&pc47,&&pc48,&&pc49,&&pc50,&&pc51,&&pc52,&&pc53,&&pc54,&&pc55,&&pc56,&&pc57,&&pc58,&&pc59,&&pc60,&&pc61,&&pc62,&&pc63,&&pc64,&&pc65,&&pc66,&&pc67,&&pc68,&&pc69,&&pc70,&&pc71,&&pc72,&&pc73,&&pc74,&&pc75,&&pc76,&&pc77,&&pc78,&&pc79,&&pc80,&&pc81,&&pc82,&&pc83,&&pc84,&&pc85,&&pc86,&&pc87,&&pc88,&&pc89,&&pc90,&&pc91,&&pc92,&&pc93,&&pc94,&&pc95,&&pc96,&&pc97,&&pc98,&&pc99,&&pc100};
int *pc=code+84, *sp = stack+10000, *bp=sp, a, *t;
goto *_pc[pc-code];
pc0:pc++; /*ENT 2*/ pc1:*--sp = (int)bp; bp = sp; sp = sp - *pc++;
pc2:pc++; /*LVAR -1*/ pc3:a = (int)(bp + *pc++);
pc4:pc++; /*PSH*/ *--sp = a;
pc5:pc++; /*IMM 0*/ pc6:a = *pc++;
pc7:pc++; /*SI*/ *(int *)*sp++ = a;
pc8:pc++; /*LVAR -2*/ pc9:a = (int)(bp + *pc++);
pc10:pc++; /*PSH*/ *--sp = a;
pc11:pc++; /*IMM 1*/ pc12:a = *pc++;
pc13:pc++; /*SI*/ *(int *)*sp++ = a;
pc14:pc++; /*STR 0*/ pc15:a = (int)(stab+*pc++);
pc16:pc++; /*PSH*/ *--sp = a;
pc17:pc++; /*LVAR -2*/ pc18:a = (int)(bp + *pc++);
pc19:pc++; /*LI*/ a = *(int *)a;
pc20:pc++; /*PSH*/ *--sp = a;
pc21:pc++; /*LVAR -1*/ pc22:a = (int)(bp + *pc++);
pc23:pc++; /*LI*/ a = *(int *)a;
pc24:pc++; /*PSH*/ *--sp = a;
pc25:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc26:pc++; /*ADJ 3*/ pc27:sp = sp + *pc++;
pc28:pc++; /*LVAR -2*/ pc29:a = (int)(bp + *pc++);
pc30:pc++; /*LI*/ a = *(int *)a;
pc31:pc++; /*PSH*/ *--sp = a;
pc32:pc++; /*LVAR 2*/ pc33:a = (int)(bp + *pc++);
pc34:pc++; /*LI*/ a = *(int *)a;
pc35:pc++; /*LE*/ a = *sp++ <= a;
pc36:pc++; /*BZ 42*/ pc37:if (!a) { pc=(int *)(pc+*pc); goto *_pc[pc-code];} else pc++;
pc38:pc++; /*STR 12*/ pc39:a = (int)(stab+*pc++);
pc40:pc++; /*PSH*/ *--sp = a;
pc41:pc++; /*LVAR -2*/ pc42:a = (int)(bp + *pc++);
pc43:pc++; /*LI*/ a = *(int *)a;
pc44:pc++; /*PSH*/ *--sp = a;
pc45:pc++; /*LVAR -1*/ pc46:a = (int)(bp + *pc++);
pc47:pc++; /*LI*/ a = *(int *)a;
pc48:pc++; /*PSH*/ *--sp = a;
pc49:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc50:pc++; /*ADJ 3*/ pc51:sp = sp + *pc++;
pc52:pc++; /*LVAR -1*/ pc53:a = (int)(bp + *pc++);
pc54:pc++; /*PSH*/ *--sp = a;
pc55:pc++; /*LVAR -1*/ pc56:a = (int)(bp + *pc++);
pc57:pc++; /*LI*/ a = *(int *)a;
pc58:pc++; /*PSH*/ *--sp = a;
pc59:pc++; /*LVAR -2*/ pc60:a = (int)(bp + *pc++);
pc61:pc++; /*LI*/ a = *(int *)a;
pc62:pc++; /*ADD*/ a = *sp++ +  a;
pc63:pc++; /*SI*/ *(int *)*sp++ = a;
pc64:pc++; /*LVAR -2*/ pc65:a = (int)(bp + *pc++);
pc66:pc++; /*PSH*/ *--sp = a;
pc67:pc++; /*LI*/ a = *(int *)a;
pc68:pc++; /*PSH*/ *--sp = a;
pc69:pc++; /*IMM 1*/ pc70:a = *pc++;
pc71:pc++; /*ADD*/ a = *sp++ +  a;
pc72:pc++; /*SI*/ *(int *)*sp++ = a;
pc73:pc++; /*PSH*/ *--sp = a;
pc74:pc++; /*IMM 1*/ pc75:a = *pc++;
pc76:pc++; /*SUB*/ a = *sp++ -  a;
pc77:pc++; /*JMP -50*/ pc78:pc = (int *) (pc+*pc); goto *_pc[pc-code];
pc79:pc++; /*LVAR -1*/ pc80:a = (int)(bp + *pc++);
pc81:pc++; /*LI*/ a = *(int *)a;
pc82:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *_pc[pc-code];
pc83:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *_pc[pc-code];
pc84:pc++; /*ENT 0*/ pc85:*--sp = (int)bp; bp = sp; sp = sp - *pc++;
pc86:pc++; /*STR 24*/ pc87:a = (int)(stab+*pc++);
pc88:pc++; /*PSH*/ *--sp = a;
pc89:pc++; /*IMM 10*/ pc90:a = *pc++;
pc91:pc++; /*PSH*/ *--sp = a;
pc92:pc++; /*JSR -93*/ pc93:*--sp = (int)(pc + 1); pc = (int *)(pc+*pc); goto *_pc[pc-code];
pc94:pc++; /*ADJ 1*/ pc95:sp = sp + *pc++;
pc96:pc++; /*PSH*/ *--sp = a;
pc97:pc++; /*PRTF*/ t = sp + pc[1]; a = printf((char *)t[-1], t[-2], t[-3], t[-4], t[-5], t[-6]);
pc98:pc++; /*ADJ 2*/ pc99:sp = sp + *pc++;
pc100:pc++; /*LEV*/ sp = bp; bp = (int *)*sp++; pc = (int *)*sp++;if (pc==NULL) return 0; else goto *_pc[pc-code];
}
int main() {
  run();
}
