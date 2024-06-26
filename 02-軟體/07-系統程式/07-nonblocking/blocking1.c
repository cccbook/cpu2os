#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char* argv[])
{
    int fd = open("/dev/tty",O_RDONLY); //Open the standard I / O file, which is blocked.
    if(fd == -1){
        perror("open /dev/tty");
        exit(1);
    }
    int ret = 0;
    char buf[1024] = {0};
    while(1){
        ret = read(fd, buf, sizeof(buf));
        if(ret == -1){
            perror("read");
            exit(1);
        }
        else if(ret == 0)
            printf("buf is null\n");
        else if(ret > 0)
            printf("buf is %s\n",buf);
        printf("ret = %d\n", ret);
        sleep(1);
    }
    close(fd);

    return 0;
}