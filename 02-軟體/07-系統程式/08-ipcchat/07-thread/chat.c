#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <pthread.h>

#define SMAX 80
int fd;

void *receiver(void *argu) {
    char msg[SMAX];
    while (1) {
        int n = recv(fd, msg, SMAX, 0);
        if (n <=0) break;
        printf("receive: %s", msg);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    int cfd;
    struct sockaddr_in saddr, raddr;
    memset(&saddr, 0, sizeof(saddr));
    memset(&raddr, 0, sizeof(raddr));
    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(8888);
    char msg[SMAX];
    if (argc==1) { // server
        printf("I am server...\n");
        saddr.sin_addr.s_addr = INADDR_ANY;
        bind(sfd, (struct sockaddr*) &saddr, sizeof(struct sockaddr));
        listen(sfd, 1);
        socklen_t rAddrLen = sizeof(struct sockaddr);
        cfd = accept(sfd, (struct sockaddr*) &raddr, &rAddrLen);
        printf("accept: cfd=%d client addr %s\n", cfd, inet_ntoa(raddr.sin_addr));
        fd = cfd;
    } else { // client
        printf("I am client...\n");
        saddr.sin_addr.s_addr = inet_addr(argv[1]);
        connect(sfd, (struct sockaddr*) &saddr, sizeof(struct sockaddr));
        fd = sfd;
        printf("connect success: sfd=%d server addr=%s\n", sfd, inet_ntoa(saddr.sin_addr));
    }
    pthread_t thread1;     // 宣告執行緒
    pthread_create(&thread1, NULL, &receiver, NULL);   
    // readline and send msg
    while (1) {
        fgets(msg, SMAX, stdin);
        send(fd, msg, strlen(msg)+1, 0);
    }
    close(sfd);
    return 0;
}
