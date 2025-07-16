#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char* argv[])
{
    int fd = open("/dev/tty", O_RDONLY);

    //The fcntl() function sets the standard I / O file to non blocking
    int flag = fcntl(fd, F_GETFL);  //Step 1: get file operation permission
    flag |= O_NONBLOCK;       //Step 2: get the file property that is the same as o ﹣ Nonblock or
    fcntl(fd, F_SETFL, flag); //Step 3: set file operation permission again
    int ret = 0;
    char buf[1024] = {0};
    
    while(1){
        ret = read(fd, buf, sizeof(buf));
        if(ret == -1){
            perror("read /dev/tty");
            printf("no input,buf is null\n");
        }
        else {
            printf("ret = %d, buf is %s\n",ret, buf);
        }
        sleep(1);
    }
    close(fd);

    return 0;
}
