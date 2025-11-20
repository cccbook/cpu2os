#include "riscv.h"
#include "uart.h"
#include <stdint.h>

#define MAX_THREADS 5
#define THREAD_STACK_SIZE 1024
#define TIMESLICE 100000 // A simple time slice value

// Thread states
enum {
    UNUSED,
    RUNNING,
    READY,
};

// Thread Control Block (TCB)
struct tcb {
    struct context context;
    uint64_t stack[THREAD_STACK_SIZE];
    int state;
};

// Global variables
struct tcb threads[MAX_THREADS];
int current_thread = 0; // Start with thread 0

// External assembly functions
extern void timer_vector();
extern void switch_to(struct context *new_context);

// Round-robin scheduler
void schedule() {
    int next_thread = current_thread;
    for (int i = 1; i <= MAX_THREADS; i++) {
        next_thread = (current_thread + i) % MAX_THREADS;
        // Find the next available thread (not UNUSED)
        if (threads[next_thread].state != UNUSED) {
            break;
        }
    }

    if (next_thread != current_thread) {
        if (threads[current_thread].state == RUNNING) {
            threads[current_thread].state = READY;
        }
        threads[next_thread].state = RUNNING;
        current_thread = next_thread;
    }
}

// C-level entry point for the timer interrupt
void timer_handler() {
    // 1. Reset the timer for the next interrupt
    volatile uint64_t *mtimecmp = (uint64_t *)0x02004000;
    volatile uint64_t *mtime = (uint64_t *)0x0200bff8;
    *mtimecmp = *mtime + TIMESLICE;

    // 2. Call the scheduler to decide the next thread
    schedule();
}

// Create a new thread
int thread_create(void (*func)()) {
    for (int i = 0; i < MAX_THREADS; i++) {
        if (threads[i].state == UNUSED) {
            threads[i].state = READY;
            // Set the thread's entry point (mepc) and stack pointer (sp)
            threads[i].context.mepc = (uint64_t)func;
            threads[i].context.sp = (uint64_t)&threads[i].stack[THREAD_STACK_SIZE];
            return i;
        }
    }
    return -1; // Failed to create thread
}

// A simple thread function
void thread_function_1() {
    while (1) {
        uart_puts("Hello from thread 1\n");
        // Use a volatile loop to prevent compiler optimization
        for (volatile int i = 0; i < 5000000; i++);
    }
}

// Another simple thread function
void thread_function_2() {
    while (1) {
        uart_puts("Hello from thread 2\n");
        // Use a volatile loop to prevent compiler optimization
        for (volatile int i = 0; i < 5000000; i++);
    }
}

// Main function
int main() {
    uart_init();
    uart_puts("OS starting...\n");

    // Initialize all TCBs to UNUSED
    for (int i = 0; i < MAX_THREADS; i++) {
        threads[i].state = UNUSED;
    }

    // Create our threads
    thread_create(thread_function_1);
    thread_create(thread_function_2);

    // Set the first thread to run
    threads[current_thread].state = RUNNING;

    // Set up the first timer interrupt
    volatile uint64_t *mtimecmp = (uint64_t *)0x02004000;
    volatile uint64_t *mtime = (uint64_t *)0x0200bff8;
    *mtimecmp = *mtime + TIMESLICE;

    // Set the machine trap vector and enable interrupts
    w_mtvec((uint64_t)timer_vector);
    w_mstatus(r_mstatus() | MSTATUS_MIE);
    w_mie(r_mie() | MIE_MTIE);

    // Start the first thread. This function will not return.
    switch_to(&threads[current_thread].context);

    // This part should never be reached
    while (1);
    return 0;
}