#ifndef __RISCV_H__
#define __RISCV_H__

#include <stdint.h>

// Machine Status Register (mstatus) bits
#define MSTATUS_MIE (1 << 3) // Machine-mode Interrupt Enable

// Machine Interrupt Enable Register (mie) bits
#define MIE_MTIE (1 << 7) // Machine Timer Interrupt Enable

// Full context that needs to be saved for a thread
struct context {
    uint64_t ra;   // x1: Return Address
    uint64_t sp;   // x2: Stack Pointer
    uint64_t gp;   // x3: Global Pointer
    uint64_t tp;   // x4: Thread Pointer
    uint64_t t0;   // x5: Temporary/alternate link register
    uint64_t t1;   // x6: Temporary
    uint64_t t2;   // x7: Temporary
    uint64_t s0;   // x8: Saved register/frame pointer
    uint64_t s1;   // x9: Saved register
    uint64_t a0;   // x10: Function argument/return value
    uint64_t a1;   // x11: Function argument/return value
    uint64_t a2;   // x12: Function argument
    uint64_t a3;   // x13: Function argument
    uint64_t a4;   // x14: Function argument
    uint64_t a5;   // x15: Function argument
    uint64_t a6;   // x16: Function argument
    uint64_t a7;   // x17: Function argument
    uint64_t s2;   // x18: Saved register
    uint64_t s3;   // x19: Saved register
    uint64_t s4;   // x20: Saved register
    uint64_t s5;   // x21: Saved register
    uint64_t s6;   // x22: Saved register
    uint64_t s7;   // x23: Saved register
    uint64_t s8;   // x24: Saved register
    uint64_t s9;   // x25: Saved register
    uint64_t s10;  // x26: Saved register
    uint64_t s11;  // x27: Saved register
    uint64_t t3;   // x28: Temporary
    uint64_t t4;   // x29: Temporary
    uint64_t t5;   // x30: Temporary
    uint64_t t6;   // x31: Temporary
    uint64_t mepc; // Machine Exception Program Counter
};


static inline uint64_t r_mstatus() {
    uint64_t x;
    __asm__ volatile("csrr %0, mstatus" : "=r" (x));
    return x;
}

static inline void w_mstatus(uint64_t x) {
    __asm__ volatile("csrw mstatus, %0" : : "r" (x));
}

static inline uint64_t r_mie() {
    uint64_t x;
    __asm__ volatile("csrr %0, mie" : "=r" (x));
    return x;
}

static inline void w_mie(uint64_t x) {
    __asm__ volatile("csrw mie, %0" : : "r" (x));
}

static inline void w_mtvec(uint64_t x) {
    __asm__ volatile("csrw mtvec, %0" : : "r" (x));
}

#endif
