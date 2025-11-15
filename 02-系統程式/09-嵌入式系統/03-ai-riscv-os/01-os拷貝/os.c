#include "riscv.h"
#include "uart.h"
#include <stdint.h>

#define MAX_THREADS 5
#define THREAD_STACK_SIZE 1024
#define TIMESLICE 100000

// Thread states
enum {
    UNUSED,
    RUNNING,
    READY,
};

// Thread context
struct context {
    uint64_t ra;
    uint64_t sp;
    uint64_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
};

// Thread control block
struct tcb {
    struct context context;
    uint64_t stack[THREAD_STACK_SIZE];
    int state;
};

struct tcb threads[MAX_THREADS];
int current_thread = -1;

extern void switch_context(struct context *old, struct context *new);
extern void timer_vector();

// Simple scheduler
void schedule() {
    int next_thread = -1;

    for (int i = 0; i < MAX_THREADS; i++) {
        if (threads[i].state == READY && i != current_thread) {
            next_thread = i;
            break;
        }
    }

    if (next_thread == -1) {
        if (current_thread != -1 && threads[current_thread].state == RUNNING) {
            return; // No other ready threads
        }
        for (int i = 0; i < MAX_THREADS; i++) {
            if (threads[i].state == READY) {
                next_thread = i;
                break;
            }
        }
    }

    if (next_thread != -1) {
        int old_thread = current_thread;
        current_thread = next_thread;

        threads[old_thread].state = READY;
        threads[current_thread].state = RUNNING;

        switch_context(&threads[old_thread].context, &threads[current_thread].context);
    }
}

// Timer interrupt handler
void timer_handler() {
    // Set the next timer interrupt
    volatile uint64_t *mtimecmp = (uint64_t *)0x02004000;
    volatile uint64_t *mtime = (uint64_t *)0x0200bff8;
    *mtimecmp = *mtime + TIMESLICE;

    schedule();
}

// Create a new thread
int thread_create(void (*func)()) {
    for (int i = 0; i < MAX_THREADS; i++) {
        if (threads[i].state == UNUSED) {
            threads[i].state = READY;
            threads[i].context.ra = (uint64_t)func;
            threads[i].context.sp = (uint64_t)&threads[i].stack[THREAD_STACK_SIZE - 1];
            return i;
        }
    }
    return -1; // No free TCB
}

// A simple thread function
void thread_function_1() {
    while (1) {
        uart_puts("Hello from thread 1\n");
        for (int i = 0; i < 10000000; i++)
            asm volatile("nop");
    }
}

// Another simple thread function
void thread_function_2() {
    while (1) {
        uart_puts("Hello from thread 2\n");
        for (int i = 0; i < 10000000; i++)
            asm volatile("nop");
    }
}

// Main function
int main() {
    uart_init();
    uart_puts("OS starting...\n");

    // Initialize TCBs
    for (int i = 0; i < MAX_THREADS; i++) {
        threads[i].state = UNUSED;
    }

    // Create threads
    thread_create(thread_function_1);
    thread_create(thread_function_2);

    // Initial thread to start
    current_thread = 0;
    threads[current_thread].state = RUNNING;

    // Set up timer interrupt
    volatile uint64_t *mtimecmp = (uint64_t *)0x02004000;
    volatile uint64_t *mtime = (uint64_t *)0x0200bff8;
    *mtimecmp = *mtime + TIMESLICE;

    // Enable timer interrupt
    w_mtvec((uint64_t)timer_vector);
    w_mstatus(r_mstatus() | MSTATUS_MIE);
    w_mie(r_mie() | MIE_MTIE);

    // Switch to the first thread
    switch_context(&threads[MAX_THREADS - 1].context, &threads[current_thread].context);

    while (1); // Should not reach here
    return 0;
}
