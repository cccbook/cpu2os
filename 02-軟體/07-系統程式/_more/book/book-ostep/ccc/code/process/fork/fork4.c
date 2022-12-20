#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char *argv[]) {
    int id = fork();
    if (id < 0) { // fork failed; exit
        fprintf(stderr, "fork failed\n");
        exit(1);
    } else if (id == 0) { // child (new process)
        printf("output to fork4.output\n");
        // child: redirect standard output to a file
        close(STDOUT_FILENO);
        open("./fork4.output", O_CREAT|O_WRONLY|O_TRUNC, S_IRWXU);
        char *myargs[3] = { "wc", "fork4.c", NULL };
        execvp(myargs[0], myargs);  // runs word count
    } else { // parent goes down this path (original process)
        wait(NULL);
    }
    return 0;
}
