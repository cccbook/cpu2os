#include <stdio.h>
#include <term.h>
#include <ncurses.h>

int main() {
    int nrows, ncols;

    setupterm(NULL, fileno(stdout), (int *)0);
    nrows = tigetnum("lines");
    ncols = tigetnum("cols");
    printf("nrows=%d ncolumns=%d\n", nrows, ncols);
}
