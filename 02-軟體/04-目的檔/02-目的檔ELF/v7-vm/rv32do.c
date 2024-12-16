#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lib.h"

// 取得 I 型指令
static void do_i_type(uint32_t instr)
{
    uint32_t opcode = instr & 0x7F;
    uint32_t rd = (instr >> 7) & 0x1F;     // 目的暫存器
    uint32_t funct3 = (instr >> 12) & 0x7; // 功能碼
    uint32_t rs1 = (instr >> 15) & 0x1F;   // 原暫存器1
    int32_t imm = (instr >> 20);           // 立即數，符號擴展
    imm = sign_extend_12(imm);

    if (opcode == 0x67)
    {
        jalr(rd, rs1, imm);
        return;
    }
    else if (opcode == 0x73)
    {
        if (imm == 0)
        {
            ecall();
        }
        else if (imm == 1)
        {
            ebreak();
        }
        return;
    }
    else if (opcode == 0x03)
    {
        switch (funct3)
        {
        case 0x0: // lb
            lb(rd, rs1, imm);
            break;
        case 0x1: // lh
            lh(rd, rs1, imm);
            break;
        case 0x2: // lw
            lw(rd, rs1, imm);
            break;
        case 0x4: // lbu
            lbu(rd, rs1, imm);
            break;
        case 0x5: // lhu
            lhu(rd, rs1, imm);
            break;
        default:
            error("Unknown I-Type");
            break;
        }
    }
    else if (opcode == 0x13)
    {
        switch (funct3)
        {
        case 0x0:
            addi(rd, rs1, imm);
            break;
        case 0x2: // slti
            slti(rd, rs1, imm);
            break;
        case 0x3: // sltiu
            sltiu(rd, rs1, imm);
            break;
        case 0x4: // xori
            xori(rd, rs1, imm);
            break;
        case 0x6: // ori
            ori(rd, rs1, imm);
            break;
        case 0x7: // andi
            andi(rd, rs1, imm);
            break;
        default:
            error("Unknown I-Type");
            break;
        }
    }
}

// 取得 Branch 型指令
static void do_b_type(uint32_t instr)
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
        beq(rs1, rs2, imm);
        break;
    case 0x1: // BNE
        bne(rs1, rs2, imm);
        break;
    case 0x4: // BLT
        blt(rs1, rs2, imm);
        break;
    case 0x5: // BGE
        bge(rs1, rs2, imm);
        break;
    case 0x6: // BLTU
        bltu(rs1, rs2, imm);
        break;
    case 0x7: // BGEU
        bgeu(rs1, rs2, imm);
        break;
    default:
        error("Unknown B-Type");
        break;
    }
}

// 取得 R 型指令
static void do_r_type(uint32_t instr)
{
    uint32_t rd = (instr >> 7) & 0x1F;
    uint32_t funct3 = (instr >> 12) & 0x7;
    uint32_t rs1 = (instr >> 15) & 0x1F;
    uint32_t rs2 = (instr >> 20) & 0x1F;
    uint32_t funct7 = (instr >> 25) & 0x7F;

    if (funct7 == 0x00 && funct3 == 0x00) 
    {
        add(rd, rs1, rs2);
    }
    else if (funct7 == 0x20 && funct3 == 0x00)
    {
        sub(rd, rs1, rs2);
    }
    else if (funct3 == 0x01)
    {
        sll(rd, rs1, rs2);
    }
    else if (funct3 == 0x02)
    {
        slt(rd, rs1, rs2);
    }
    else if (funct3 == 0x03)
    {
        sltu(rd, rs1, rs2);
    }
    else if (funct3 == 0x04)
    {
        xor(rd, rs1, rs2);
    }
    else if (funct3 == 0x05 && funct7 == 0x00)
    {
        srl(rd, rs1, rs2);
    }
    else if (funct3 == 0x05 && funct7 == 0x20)
    {
        sra(rd, rs1, rs2);
    }
    else if (funct3 == 0x06)
    {
        or (rd, rs1, rs2);
    }
    else if (funct3 == 0x07)
    {
        and(rd, rs1, rs2);
    }
    else
        error("Unknown R-Type");
}

// 取得 S 型指令
static void do_s_type(uint32_t instr)
{
    uint32_t funct3 = (instr >> 12) & 0x7;
    uint32_t rs1 = (instr >> 15) & 0x1F;
    uint32_t rs2 = (instr >> 20) & 0x1F;
    int32_t imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F);
    imm = sign_extend_12(imm);

    if (funct3 == 0x0)
    {
        sb(rs1, rs2, imm);
    }
    else if (funct3 == 0x1)
    {
        sh(rs1, rs2, imm);
    }
    else if (funct3 == 0x2)
    {
        sw(rs1, rs2, imm);
    }
    else
        error("Unknown S-Type");
}

// 取得 U 型指令
static void do_u_type(uint32_t instr)
{
    uint32_t opcode = instr & 0x7F;
    uint32_t rd = (instr >> 7) & 0x1F;
    int32_t imm = (instr & 0xFFFFF000) >> 12; // 高20位立即數

    if (opcode == 0x37)
    {
        lui(rd, imm);
    }
    else if (opcode == 0x17)
    {
        auipc(rd, imm);
    }
    else
    {
        error("Unknown U-Type");
    }
}

// 取得j 型指令
static void do_j_type(uint32_t instr)
{
    uint32_t rd = (instr >> 7) & 0x1F;
    int32_t imm = ((instr >> 12) & 0xFF) << 12 |
                  ((instr >> 20) & 1) << 11 |
                  ((instr >> 21) & 0x3FF) << 1;
    imm = sign_extend_20(imm);
    jal(rd, imm);
}

// 根據指令類型調用對應的反組譯函數
static char do_instr(uint32_t instr)
{
    uint32_t opcode = instr & 0x7F;
    char type = '?';
    switch (opcode)
    {
    case 0x03: // Load
    case 0x13: // I-Type
    case 0x67: // JALR
    case 0x73: // ECALL
        do_i_type(instr);
        type = 'I';
        break;
    case 0x33: // R-Type
        do_r_type(instr);
        type = 'R';
        break;
    case 0x23: // S-Type
        do_s_type(instr);
        type = 'S';
        break;
    case 0x37: // LUI
    case 0x17: // AUIPC
        do_u_type(instr);
        type = 'U';
        break;
    case 0x63: // B 型指令
        do_b_type(instr);
        type = 'B';
        break;
    case 0x6F: // JAL
        type = 'J';
        do_j_type(instr);
        break;
    default:
        error("Unknown instruction");
    }
    return type;
}
/*
void do_block(char *block, int size)
{
    for (int pc = 0; pc < size; pc += 4)
    {
        uint32_t instr = decode_little_endian32(&block[pc]);
        char type = do_instr(instr);
        printf("%04x %08x %c %s\n", pc, instr, type, oasm);
    }
}
*/
