char asm1[100];

#define dasm(...) sprintf(asm1, __VA_ARGS__)

#define ecall() dasm("ecall")
#define ebreak() dasm("ebreak")

#define jalr(rd, rs, imm) dasm("jalr x%d, x%d, %d", rd, rs, imm)

#define lb(rd, rs, imm)   dasm("lb x%d, x%d, %d", rd, rs, imm)
#define lh(rd, rs, imm)   dasm("lh x%d, x%d, %d", rd, rs, imm)
#define lw(rd, rs, imm)   dasm("lw x%d, x%d, %d", rd, rs, imm)
#define lbu(rd, rs, imm)  dasm("lbu x%d, x%d, %d", rd, rs, imm)
#define lhu(rd, rs, imm)  dasm("lhu x%d, x%d, %d", rd, rs, imm)

#define addi(rd, rs, imm) dasm("addi x%d, x%d, %d", rd, rs, imm)
#define slti(rd, rs, imm) dasm("slti x%d, x%d, %d", rd, rs, imm)
#define sltiu(rd, rs, imm) dasm("sltiu x%d, x%d, %d", rd, rs, imm)
#define xori(rd, rs, imm) dasm("xori x%d, x%d, %d", rd, rs, imm)
#define ori(rd, rs, imm)  dasm("ori x%d, x%d, %d", rd, rs, imm)
#define andi(rd, rs, imm) dasm("andi x%d, x%d, %d", rd, rs, imm)

#define beq(rs1, rs2, imm) dasm("beq x%d, x%d, %d", rs1, rs2, imm)
#define bne(rs1, rs2, imm) dasm("bne x%d, x%d, %d", rs1, rs2, imm)
#define blt(rs1, rs2, imm) dasm("blt x%d, x%d, %d", rs1, rs2, imm)
#define bge(rs1, rs2, imm) dasm("bge x%d, x%d, %d", rs1, rs2, imm)
#define bltu(rs1, rs2, imm) dasm("bltu x%d, x%d, %d", rs1, rs2, imm)
#define bgeu(rs1, rs2, imm) dasm("bgeu x%d, x%d, %d", rs1, rs2, imm)

#define add(rd, rs1, rs2) dasm("add x%d, x%d, x%d", rd, rs1, rs2)
#define sub(rd, rs1, rs2) dasm("sub x%d, x%d, x%d", rd, rs1, rs2)
#define sll(rd, rs1, rs2) dasm("sll x%d, x%d, x%d", rd, rs1, rs2)
#define slt(rd, rs1, rs2) dasm("slt x%d, x%d, x%d", rd, rs1, rs2)
#define sltu(rd, rs1, rs2) dasm("sltu x%d, x%d, x%d", rd, rs1, rs2)
#define xor(rd, rs1, rs2) dasm("xor x%d, x%d, x%d", rd, rs1, rs2)
#define srl(rd, rs1, rs2) dasm("srl x%d, x%d, x%d", rd, rs1, rs2)
#define sra(rd, rs1, rs2) dasm("sra x%d, x%d, x%d", rd, rs1, rs2)
#define or(rd, rs1, rs2) dasm("or x%d, x%d, x%d", rd, rs1, rs2)
#define and(rd, rs1, rs2) dasm("and x%d, x%d, x%d", rd, rs1, rs2)

#define sb(rs1, rs2, imm) dasm("sb x%d, %d(x%d)", rs1, imm, rs2)
#define sh(rs1, rs2, imm) dasm("sh x%d, %d(x%d)", rs1, imm, rs2)
#define sw(rs1, rs2, imm) dasm("sw x%d, %d(x%d)", rs1, imm, rs2)

#define lui(rd, imm) dasm("lui x%d, %d", rd, imm)
#define auipc(rd, imm) dasm("auipc x%d, %d", rd, imm)

#define jal(rd, imm) dasm("jal x%d, %d", rd, imm)

#include "rv32do.c"

void disassemble_block(char *block, int size)
{
    for (int pc = 0; pc < size; pc += 4)
    {
        uint32_t instruction = decode_little_endian32(&block[pc]);
        char asm1[100];
        char type = do_instr(instruction, asm1);
        printf("%04x %08x %c %s\n", pc, instruction, type, asm1);
    }
}
