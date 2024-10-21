#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MEM_SIZE 10000
int32_t pc, pc_new;
int32_t reg[32];
int32_t memory[MEM_SIZE];

// 取得 I 型指令
void exec_i_type(uint32_t instr)
{
    uint32_t opcode = instr & 0x7F;
    uint32_t rd = (instr >> 7) & 0x1F;     // 目的暫存器
    uint32_t funct3 = (instr >> 12) & 0x7; // 功能碼
    uint32_t rs1 = (instr >> 15) & 0x1F;   // 原暫存器1
    int32_t imm = (instr >> 20);           // 立即數，符號擴展
    imm = sign_extend_12(imm);

    if (opcode == 0x67)
    {
        printf("jalr x%d, x%d, %d", rd, rs1, imm);
        return;
    }
    else if (opcode == 0x03)
    {
        switch (funct3)
        {
        case 0x0: // lb
            printf("lb x%d, %d(x%d)", rd, imm, rs1);
            break;
        case 0x1: // lh
            printf("lh x%d, %d(x%d)", rd, imm, rs1);
            break;
        case 0x2: // lw
            printf("lw x%d, %d(x%d)", rd, imm, rs1);
            break;
        case 0x4: // lbu
            printf("lbu x%d, %d(x%d)", rd, imm, rs1);
            break;
        case 0x5: // lhu
            printf("lhu x%d, %d(x%d)", rd, imm, rs1);
            break;
        default:
            printf("Unknown I-Type");
            break;
        }
    }
    else if (opcode == 0x13)
    {
        switch (funct3)
        {
        case 0x0:
            printf("addi x%d, x%d, %d", rd, rs1, imm);
            reg[rd] = reg[rs1]+imm;
            break;
        case 0x2: // slti
            printf("slti x%d, x%d, %d", rd, rs1, imm);
            break;
        case 0x3: // sltiu
            printf("sltiu x%d, x%d, %d", rd, rs1, imm);
            break;
        case 0x4: // xori
            printf("xori x%d, x%d, %d", rd, rs1, imm);
            break;
        case 0x6: // ori
            printf("ori x%d, x%d, %d", rd, rs1, imm);
            break;
        case 0x7: // andi
            printf("andi x%d, x%d, %d", rd, rs1, imm);
            break;
        default:
            printf("Unknown I-Type");
            break;
        }
    }
}

// 取得 Branch 型指令
void exec_b_type(uint32_t instr)
{
    uint32_t funct3 = (instr >> 12) & 0x7;
    uint32_t rs1 = (instr >> 15) & 0x1F;
    uint32_t rs2 = (instr >> 20) & 0x1F;
    int32_t imm = ((instr >> 31) & 1) << 12 |
                  ((instr >> 7) & 1) << 11 |
                  ((instr >> 25) & 0x3F) << 5 |
                  ((instr >> 8) & 0xF) << 1; // 符號擴展
    imm = sign_extend_12(imm);

    switch (funct3)
    {
    case 0x0: // BEQ
        printf("beq x%d, x%d, %d", rs1, rs2, imm);
        break;
    case 0x1: // BNE
        printf("bne x%d, x%d, %d", rs1, rs2, imm);
        break;
    case 0x4: // BLT
        printf("blt x%d, x%d, %d", rs1, rs2, imm);
        break;
    case 0x5: // BGE
        printf("bge x%d, x%d, %d", rs1, rs2, imm);
        break;
    case 0x6: // BLTU
        printf("bltu x%d, x%d, %d", rs1, rs2, imm);
        break;
    case 0x7: // BGEU
        printf("bgeu x%d, x%d, %d", rs1, rs2, imm);
        break;
    default:
        printf("Unknown B-Type");
        break;
    }
}

// 取得 R 型指令
void exec_r_type(uint32_t instr)
{
    uint32_t rd = (instr >> 7) & 0x1F;
    uint32_t funct3 = (instr >> 12) & 0x7;
    uint32_t rs1 = (instr >> 15) & 0x1F;
    uint32_t rs2 = (instr >> 20) & 0x1F;
    uint32_t funct7 = (instr >> 25) & 0x7F;

    if (funct7 == 0x00 && funct3 == 0x00)
    {
        printf("add x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct7 == 0x20 && funct3 == 0x00)
    {
        printf("sub x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x01)
    {
        printf("sll x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x02)
    {
        printf("slt x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x03)
    {
        printf("sltu x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x04)
    {
        printf("xor x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x05 && funct7 == 0x00)
    {
        printf("srl x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x05 && funct7 == 0x20)
    {
        printf("sra x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x06)
    {
        printf("or x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x07)
    {
        printf("and x%d, x%d, x%d", rd, rs1, rs2);
    }
    else
    {
        printf("Unknown R-Type");
    }
}

// 取得 S 型指令
void exec_s_type(uint32_t instr)
{
    uint32_t funct3 = (instr >> 12) & 0x7;
    uint32_t rs1 = (instr >> 15) & 0x1F;
    uint32_t rs2 = (instr >> 20) & 0x1F;
    int32_t imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F);
    imm = sign_extend_12(imm);

    if (funct3 == 0x0)
    {
        printf("sb x%d, %d(x%d)", rs2, imm, rs1);
    }
    else if (funct3 == 0x1)
    {
        printf("sh x%d, %d(x%d)", rs2, imm, rs1);
    }
    else if (funct3 == 0x2)
    {
        memory[reg[rs1]+imm] = rs2;
        printf("sw x%d, %d(x%d)", rs2, imm, rs1);
    }
    else
    {
        printf("Unknown S-Type");
    }
}

// 取得 U 型指令
void exec_u_type(uint32_t instr)
{
    uint32_t opcode = instr & 0x7F;
    uint32_t rd = (instr >> 7) & 0x1F;
    int32_t imm = (instr & 0xFFFFF000) >> 12; // 高20位立即數

    if (opcode == 0x37)
    { // lui
        printf("lui x%d, %d", rd, imm);
    }
    else if (opcode == 0x17)
    { // auipc
        printf("auipc x%d, %d", rd, imm);
    }
    else
    {
        printf("Unknown U-Type");
    }
}

// 取得j 型指令
void exec_j_type(uint32_t instr)
{
    uint32_t rd = (instr >> 7) & 0x1F;
    int32_t imm = ((instr >> 12) & 0xFF) << 12 |
                  ((instr >> 20) & 1) << 11 |
                  ((instr >> 21) & 0x3FF) << 1;
    imm = sign_extend_20(imm);

    printf("jal x%d, %d", rd, imm);
}

// 根據指令類型調用對應的反組譯函數
void exec(uint32_t instr)
{
    uint32_t opcode = instr & 0x7F;
    uint32_t rd = (instr >> 7) & 0x1F;
    uint32_t rs1 = (instr >> 15) & 0x1F;
    uint32_t rs2 = (instr >> 20) & 0x1F;
    switch (opcode)
    {
    case 0x03: // Load
    case 0x13: // I-Type instructions
    case 0x67: // JALR
        exec_i_type(instr);
        break;
    case 0x33: // R-Type instrs
        exec_r_type(instr);
        break;
    case 0x23: // S-Type instrs
        exec_s_type(instr);
        break;
    case 0x37: // LUI
    case 0x17: // AUIPC
        exec_u_type(instr);
        break;
    case 0x63: // B 型指令
        exec_b_type(instr);
        break;
    case 0x6F: // JAL
        exec_j_type(instr);
        break;
    default:
        printf("Unknown instruction");
    }
    printf(" \tx[%d]=%d\n", rd, reg[rd]);
}

void vm_run(char *memory, int size, int entry)
{
    pc = entry;
    while (pc < size)
    {
        uint32_t instr = decode_little_endian32(&memory[pc]);
        pc_new = -1;
        printf("%04x ", pc);
        exec(instr);

        pc = (pc_new == -1)?pc+4:pc_new;
    }
}
