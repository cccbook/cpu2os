#include "obj.h"

// #define csrName(csr) "csr"#csr

void ir2asm(uint32_t ir, char *line) {
  decode(ir);
  switch (type) {
    case 'R': sprintf(line, "%s x%d,x%d,x%d", name, rd, rs1, rs2); break;
    case 'I':
      switch (op) {
        case 0x03: sprintf(line, "%s x%d,%d(x%d)", name, rd, imm, rs1); break;
        case 0x73: assert(0); /* no support for csr* command*/ break;
        default: sprintf(line, "%s x%d,x%d,%d", name, rd, rs1, imm); break;
      }
      break;
    case 'S': sprintf(line, "%s x%d,%d(x%d)", name, rs1, imm, rs2); break;
    case 'B': sprintf(line, "%s x%d,x%d,%d", name, rs1, rs2, imm); break;
    case 'U': sprintf(line, "%s x%d,%d", name, rd, imm); break;
    case 'J': sprintf(line, "%s %d(x%d)", name, imm, rd); break;
    default: sprintf(line, "Instruction error !");
  }
}

int objDump(char *objFile) {
  riscvInit();
  char sasm[LMAX];
  FILE *ofd = fopen(objFile, "rb");
  mTop = fread(m, sizeof(uint8_t), MMAX, ofd);
  fclose(ofd);
  for (pc = 0; pc<mTop; pc+=4) {
    ir = bytes2word(&m[pc], LittleEndian);
    ir2asm(ir, sasm);
    printf("%04X:%08X %s\n", pc, ir, sasm);
  }
  return 0;
}

