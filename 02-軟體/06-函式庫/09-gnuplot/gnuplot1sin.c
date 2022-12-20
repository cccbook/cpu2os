#include <stdlib.h>
#include <stdio.h>
#define NUM_POINTS 5
#define NUM_COMMANDS 2

int main()
{
    char * commandsForGnuplot[] = {"set title \"sin(x)\"", "plot sin(x)"};

    FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");

    for (int i=0; i < NUM_COMMANDS; i++)
    {
        fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); //Send commands to gnuplot one by one.
    }
    return 0;
}
