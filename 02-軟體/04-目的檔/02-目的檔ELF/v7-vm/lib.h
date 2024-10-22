#pragma once

#include <stdint.h>

#define error(msg) perror(msg)

uint32_t decode_little_endian32(const char *bytes);

// 符號擴展 12 位數
int32_t sign_extend_12(int32_t imm);

// 符號擴展 20 位數
int32_t sign_extend_20(int32_t imm);
