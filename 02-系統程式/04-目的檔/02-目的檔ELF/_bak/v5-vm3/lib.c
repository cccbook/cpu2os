#define error(msg) perror(msg)

uint32_t decode_little_endian32(const char *bytes) {
    // 根據 little-endian 方式組合 4 個字節
    uint32_t result = 0;

    result |= (uint32_t)(unsigned char)bytes[0]; // 取第 1 個字節
    result |= (uint32_t)(unsigned char)bytes[1] << 8; // 取第 2 個字節，左移 8 位
    result |= (uint32_t)(unsigned char)bytes[2] << 16; // 取第 3 個字節，左移 16 位
    result |= (uint32_t)(unsigned char)bytes[3] << 24; // 取第 4 個字節，左移 24 位

    return result;
}

// 符號擴展 12 位數
int32_t sign_extend_12(int32_t imm)
{
    return (imm & 0x800) ? (imm | 0xFFFFF000) : imm;
}

// 符號擴展 20 位數
int32_t sign_extend_20(int32_t imm)
{
    return (imm & 0x80000) ? (imm | 0xFFF00000) : imm;
}
