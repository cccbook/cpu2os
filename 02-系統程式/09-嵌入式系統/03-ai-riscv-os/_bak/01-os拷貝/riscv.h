#ifndef __RISCV_H__
#define __RISCV_H__

#include <stdint.h>

// mstatus and mie CSRs
#define MSTATUS_MIE (1 << 3)
#define MIE_MTIE (1 << 7)

// Read mstatus CSR
static inline uint64_t r_mstatus() {
    uint64_t x;
    asm volatile("csrr %0, mstatus" : "=r" (x));
    return x;
}

// Write mstatus CSR
static inline void w_mstatus(uint64_t x) {
    asm volatile("csrw mstatus, %0" : : "r" (x));
}

// Read mie CSR
static inline uint64_t r_mie() {
    uint64_t x;
    asm volatile("csrr %0, mie" : "=r" (x));
    return x;
}

// Write mie CSR
static inline void w_mie(uint64_t x) {
    asm volatile("csrw mie, %0" : : "r" (x));
}

// Write mtvec CSR (Machine Trap Vector)
static inline void w_mtvec(uint64_t x) {
    asm volatile("csrw mtvec, %0" : : "r" (x));
}

// Read mcause CSR (Machine Cause)
static inline uint64_t r_mcause() {
    uint64_t x;
    asm volatile("csrr %0, mcause" : "=r" (x));
    return x;
}

#endif
