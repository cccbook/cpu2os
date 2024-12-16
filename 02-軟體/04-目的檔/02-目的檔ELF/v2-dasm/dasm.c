#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// 取得 I 型指令
void disassemble_i_type(uint32_t instr, char *asm1)
{
    uint32_t opcode = instr & 0x7F;
    uint32_t rd = (instr >> 7) & 0x1F;     // 目的暫存器
    uint32_t funct3 = (instr >> 12) & 0x7; // 功能碼
    uint32_t rs1 = (instr >> 15) & 0x1F;   // 原暫存器1
    int32_t imm = (instr >> 20);           // 立即數，符號擴展
    imm = sign_extend_12(imm);

    if (opcode == 0x67)
    {
        sprintf(asm1, "jalr x%d, x%d, %d", rd, rs1, imm);
        return;
    }
    else if (opcode == 0x03)
    {
        switch (funct3)
        {
        case 0x0: // lb
            sprintf(asm1, "lb x%d, %d(x%d)", rd, imm, rs1);
            break;
        case 0x1: // lh
            sprintf(asm1, "lh x%d, %d(x%d)", rd, imm, rs1);
            break;
        case 0x2: // lw
            sprintf(asm1, "lw x%d, %d(x%d)", rd, imm, rs1);
            break;
        case 0x4: // lbu
            sprintf(asm1, "lbu x%d, %d(x%d)", rd, imm, rs1);
            break;
        case 0x5: // lhu
            sprintf(asm1, "lhu x%d, %d(x%d)", rd, imm, rs1);
            break;
        default:
            sprintf(asm1, "Unknown I-Type");
            break;
        }
    }
    else if (opcode == 0x13)
    {
        switch (funct3)
        {
        case 0x0: // addi 或 jalr
            sprintf(asm1, "addi x%d, x%d, %d", rd, rs1, imm);
            break;
        case 0x2: // slti
            sprintf(asm1, "slti x%d, x%d, %d", rd, rs1, imm);
            break;
        case 0x3: // sltiu
            sprintf(asm1, "sltiu x%d, x%d, %d", rd, rs1, imm);
            break;
        case 0x4: // xori
            sprintf(asm1, "xori x%d, x%d, %d", rd, rs1, imm);
            break;
        case 0x6: // ori
            sprintf(asm1, "ori x%d, x%d, %d", rd, rs1, imm);
            break;
        case 0x7: // andi
            sprintf(asm1, "andi x%d, x%d, %d", rd, rs1, imm);
            break;
        default:
            sprintf(asm1, "Unknown I-Type");
            break;
        }
    }
}

// 取得 Branch 型指令
void disassemble_b_type(uint32_t instr, char *asm1) {
    uint32_t funct3 = (instr >> 12) & 0x7;
    uint32_t rs1 = (instr >> 15) & 0x1F;
    uint32_t rs2 = (instr >> 20) & 0x1F;
    int32_t imm = ((instr >> 31) & 1) << 12 |
                  ((instr >> 7) & 1) << 11 |
                  ((instr >> 25) & 0x3F) << 5 |
                  ((instr >> 8) & 0xF) << 1; // 符號擴展
    imm = sign_extend_12(imm);

    switch (funct3) {
        case 0x0: // BEQ
            sprintf(asm1, "beq x%d, x%d, %d", rs1, rs2, imm);
            break;
        case 0x1: // BNE
            sprintf(asm1, "bne x%d, x%d, %d", rs1, rs2, imm);
            break;
        case 0x4: // BLT
            sprintf(asm1, "blt x%d, x%d, %d", rs1, rs2, imm);
            break;
        case 0x5: // BGE
            sprintf(asm1, "bge x%d, x%d, %d", rs1, rs2, imm);
            break;
        case 0x6: // BLTU
            sprintf(asm1, "bltu x%d, x%d, %d", rs1, rs2, imm);
            break;
        case 0x7: // BGEU
            sprintf(asm1, "bgeu x%d, x%d, %d", rs1, rs2, imm);
            break;
        default:
            sprintf(asm1, "Unknown B-Type");
            break;
    }
}

// 取得 R 型指令
void disassemble_r_type(uint32_t instr, char *asm1)
{
    uint32_t rd = (instr >> 7) & 0x1F;
    uint32_t funct3 = (instr >> 12) & 0x7;
    uint32_t rs1 = (instr >> 15) & 0x1F;
    uint32_t rs2 = (instr >> 20) & 0x1F;
    uint32_t funct7 = (instr >> 25) & 0x7F;

    if (funct7 == 0x00 && funct3 == 0x00)
    {
        sprintf(asm1, "add x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct7 == 0x20 && funct3 == 0x00)
    {
        sprintf(asm1, "sub x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x01)
    {
        sprintf(asm1, "sll x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x02)
    {
        sprintf(asm1, "slt x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x03)
    {
        sprintf(asm1, "sltu x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x04)
    {
        sprintf(asm1, "xor x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x05 && funct7 == 0x00)
    {
        sprintf(asm1, "srl x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x05 && funct7 == 0x20)
    {
        sprintf(asm1, "sra x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x06)
    {
        sprintf(asm1, "or x%d, x%d, x%d", rd, rs1, rs2);
    }
    else if (funct3 == 0x07)
    {
        sprintf(asm1, "and x%d, x%d, x%d", rd, rs1, rs2);
    }
    else
    {
        sprintf(asm1, "Unknown R-Type");
    }
}

// 取得 S 型指令
void disassemble_s_type(uint32_t instr, char *asm1)
{
    uint32_t funct3 = (instr >> 12) & 0x7;
    uint32_t rs1 = (instr >> 15) & 0x1F;
    uint32_t rs2 = (instr >> 20) & 0x1F;
    int32_t imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F);
    imm = sign_extend_12(imm);

    if (funct3 == 0x0)
    {
        sprintf(asm1, "sb x%d, %d(x%d)", rs2, imm, rs1);
    }
    else if (funct3 == 0x1)
    {
        sprintf(asm1, "sh x%d, %d(x%d)", rs2, imm, rs1);
    }
    else if (funct3 == 0x2)
    {
        sprintf(asm1, "sw x%d, %d(x%d)", rs2, imm, rs1);
    }
    else
    {
        sprintf(asm1, "Unknown S-Type");
    }
}

// 取得 U 型指令
void disassemble_u_type(uint32_t instr, char *asm1)
{
    uint32_t opcode = instr & 0x7F;
    uint32_t rd = (instr >> 7) & 0x1F;
    int32_t imm = (instr & 0xFFFFF000)>>12; // 高20位立即數

    if (opcode == 0x37)
    { // lui
        sprintf(asm1, "lui x%d, %d", rd, imm);
    }
    else if (opcode == 0x17)
    { // auipc
        sprintf(asm1, "auipc x%d, %d", rd, imm);
    }
    else
    {
        sprintf(asm1, "Unknown U-Type");
    }
}

// 取得 J 型指令
void disassemble_j_type(uint32_t instr, char *asm1)
{
    uint32_t rd = (instr >> 7) & 0x1F;
    int32_t imm = ((instr >> 12) & 0xFF) << 12 |
                  ((instr >> 20) & 1) << 11 |
                  ((instr >> 21) & 0x3FF) << 1;
    imm = sign_extend_20(imm);

    sprintf(asm1, "jal x%d, %d", rd, imm);
}

// 根據指令類型調用對應的反組譯函數
char disassemble(uint32_t instruction, char *asm1)
{
    uint32_t opcode = instruction & 0x7F;
    char type = '?';
    switch (opcode)
    {
    case 0x03: // Load
    case 0x13: // I-Type instructions
    case 0x67: // JALR
        disassemble_i_type(instruction, asm1);
        type = 'I';
        break;
    case 0x33: // R-Type instructions
        disassemble_r_type(instruction, asm1);
        type = 'R';
        break;
    case 0x23: // S-Type instructions
        disassemble_s_type(instruction, asm1);
        type = 'S';
        break;
    case 0x37: // LUI
    case 0x17: // AUIPC
        disassemble_u_type(instruction, asm1);
        type = 'U';
        break;
    case 0x63: // B 型指令
        disassemble_b_type(instruction, asm1);
        type = 'B';
        break;
    case 0x6F: // JAL
        type = 'J';
        disassemble_j_type(instruction, asm1);
        break;
    default:
        sprintf(asm1, "Unknown instruction");
    }
    return type;
}
