// uart.c
#include "uart.h"
#include <stdint.h>

#define UART0 0x10000000L

void uart_init() {
    // Basic UART initialization
}

void uart_puts(char *s) {
    volatile char *tx = (volatile char *)UART0;
    while (*s) {
        *tx = *s++;
    }
}
