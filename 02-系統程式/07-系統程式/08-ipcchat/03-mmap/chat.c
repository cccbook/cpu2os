#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define SMAX 80

int main(int argc, char *argv[]) {
    int id = argv[1][0]-'0';
    int fd = open("chat.dat", O_RDWR | O_CREAT);
    char *buf = mmap(NULL, 2*SMAX, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    char *myMsg, *yourMsg;
    if (id == 0) {
        myMsg = buf;
        yourMsg = buf + SMAX;
    } else {
        myMsg = buf + SMAX;
        yourMsg = buf;
    }
    if (fork() == 0) {
        // child: receive message and print
        while (1) {
            if (yourMsg[0] != '\0') {
                printf("receive: %s", yourMsg);
                yourMsg[0] = '\0';
            }
        }
    } else {
        // parent: readline and put into myMsg in buf
        while (1) {
            fgets(myMsg, SMAX, stdin);
        }
    }
    munmap(buf, 2*SMAX);
    close(fd);
    return 0;
}
