#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define SMAX 80

int main(int argc, char *argv[]) {
    int fd;
    char *fifo0 = "/tmp/user0";
    char *fifo1 = "/tmp/user1";
    mkfifo(fifo0, 0666);
    mkfifo(fifo1, 0666);

    char *me, *you;
    if (strcmp(argv[1], "1")) { // me:0 => you:1
        me = fifo0;
        you = fifo1;
    } else { // me:1 => you:0
        me = fifo1;
        you = fifo0;
    }

    char msg[SMAX];
    if (fork() == 0) {
        // child: receive message and print (一直讀對方的訊息，讀到就印出來)
        fd = open(you, O_RDONLY); // 開啟對方管道
        while (1) {
            int n = read(fd, msg, sizeof(msg)); // 讀取對方發來的訊息
            if (n <= 0) break; // 如果管道已經關閉，就離開
            printf("receive: %s", msg); // 印出訊息
        }
        close(fd); // 關閉管道
    } else {
        // parent: readline and send （一直讀鍵盤，然後把訊息送給對方）
        fd = open(me, O_WRONLY); // 開啟我方管道
        while (1) {
            fgets(msg, SMAX, stdin); // 讀一行輸入
            int n = write(fd, msg, strlen(msg)+1); // 將該行輸入訊息送上我方管道
            if (n<=0) break;
        }
        close(fd);
    }
    return 0;
}
