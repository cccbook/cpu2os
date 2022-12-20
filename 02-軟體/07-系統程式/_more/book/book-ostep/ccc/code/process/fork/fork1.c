#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    printf("hello world (pid:%d)\n", (int) getpid());
    int id = fork();
    if (id < 0) { // fork failed; exit
        fprintf(stderr, "fork failed\n");
        exit(1);
    } else if (id == 0) { // child (new process)
        printf("hello, I am child (pid:%d)\n", (int) getpid());
    } else { // parent goes down this path (original process)
        // int cret;
        // int cid = wait(&cret); 
        printf("hello, I am parent of %d (pid:%d)\n", id, (int) getpid());
        // printf("  (cid:%d, cret=%d)\n", cid, cret);
    }
    return 0;
}
