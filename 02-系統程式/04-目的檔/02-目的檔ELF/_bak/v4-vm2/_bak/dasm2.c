#include <stdio.h>
#include <stdint.h>

// 函數用來根據給定的 32 位 RISC-V 指令返回其型態
char disassemble(uint32_t instr, char *line)
{
    uint32_t opcode = instr & 0x7F;
    uint32_t funct3 = (instr >> 12) & 0x7;
    uint32_t funct7 = (instr >> 25) & 0x7F;
    uint32_t rs1 = (instr >> 15) & 0x1F; // 第一個來源暫存器
    uint32_t rs2 = (instr >> 20) & 0x1F; // 第二個來源暫存器
    uint32_t rd = (instr >> 7) & 0x1F;   // 目的暫存器
    int32_t imm, immb, imm_i;
    imm_i = (instr >> 20); // I type 立即數

    // 根據 opcode 判斷指令型態
    switch (opcode)
    {
    case 0x37: // LUI (U)
        imm = (instr >> 12);
        sprintf(line, "LUI x%d, 0x%x", rd, imm);
        break;
    case 0x17: // AUIPC (U)
        imm = (instr >> 12);
        sprintf(line, "AUIPC x%d, 0x%x", rd, imm);
        break;
    case 0x6F: // JAL (J)
        // 提取並重組 20-bit 立即數
        imm = ((instr >> 31) & 0x1) << 19;   // imm[20]
        imm |= ((instr >> 12) & 0xFF) << 12; // imm[19:12]
        imm |= ((instr >> 20) & 0x1) << 11;  // imm[11]
        imm |= (instr >> 21) & 0x3FF;        // imm[10:1]
        imm = sign_extend_20(imm);
        sprintf(line, "JAL x%d, %d", rd, imm);
        break;
    case 0x67: // JALR (I)
        sprintf(line, "JALR x%d, %d(x%d)", rd, imm_i, rs1);
        break;
    case 0x03: // LOAD (I)
        switch (funct3)
        {
        case 0x0: // LB
            sprintf(line, "LB x%d, %d(x%d)", rd, imm_i, rs1);
            break;
        case 0x1: // LH
            sprintf(line, "LH x%d, %d(x%d)", rd, imm_i, rs1);
            break;
        case 0x2: // LW
            sprintf(line, "LW x%d, %d(x%d)", rd, imm_i, rs1);
            break;
        case 0x4: // LBU
            sprintf(line, "LBU x%d, %d(x%d)", rd, imm_i, rs1);
            break;
        case 0x5: // LHU
            sprintf(line, "LHU x%d, %d(x%d)", rd, imm_i, rs1);
            break;
        }
        break;
    case 0x63: // Branch (B)
        imm = ((instr >> 31) << 12) | ((instr & 0x7E000000) >> 20) | ((instr & 0x80) << 4) | ((instr >> 7) & 0x1E); // 符號擴展
        switch (funct3)
        {
            case 0x0:
                break;
            case 0x1:
                break;
            case 0x4:
                break;
            case 0x5:
                break;
            case 0x6:
                break;
            case 0x7:
                break;
        }
        break;
    case 0x23: // Store (S)
        imm = (((instr >> 25) & 0x7F) << 5) | ((instr >> 7) & 0x1F);
        break;
    case 0x13: // OPI (I)
        imm = imm_i;
        break;
    case 0x33: // OP (R)
        break;
    case 0x0F: // FENCE (I)
        break;
    case 0x73: // CSR (I)
        break;
    default:
        break;
    }
}
