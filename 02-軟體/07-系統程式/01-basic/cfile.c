#include <stdio.h>
// #include <unistd.h>
#include <assert.h>
// #include <fcntl.h>
// #include <sys/stat.h>
// #include <sys/types.h>
#include <string.h>

int main(int argc, char *argv[]) {
    FILE * fd = fopen("hello.txt", "w+");
    assert(fd != NULL);
    char buffer[20];
    sprintf(buffer, "hello world!\n");
    int rc = fwrite(buffer, sizeof(char), strlen(buffer), fd);
    assert(rc == (strlen(buffer)));
    fflush(fd); // fsync(fd);
    fclose(fd);
    return 0;
}

