#include <stdio.h>
#include <assert.h>
#include <string.h>
// #define __POSIX__
#ifdef __POSIX__
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifdef __WINDOWS__
// ....
#endif

int main(int argc, char *argv[]) {
    int fd = open("hello.txt", O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    assert(fd >= 0);
    char buffer[20];
    sprintf(buffer, "hello world!\n");
    int rc = write(fd, buffer, strlen(buffer));
    assert(rc == (strlen(buffer)));
    fsync(fd);
    close(fd);
    return 0;
}

