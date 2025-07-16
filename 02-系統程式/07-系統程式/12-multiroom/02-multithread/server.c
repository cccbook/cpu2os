#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <pthread.h>
#include <assert.h>

#define SMAX 80
#define CMAX 10 // 100000

struct Client_t { // 客戶端的資料結構
    int fd; // 串流代號
    pthread_t thread; // 線程 thread
};

struct Client_t clients[CMAX]; // 所有客戶端

void *receiver(void *argu) {
    int ci = *(int*)argu;
    int cfd = clients[ci].fd;
    char msg[SMAX];
    while (1) {
        int n = recv(cfd, msg, SMAX, 0); // 收到某客戶端傳來的訊息
        if (n <=0) break;
        printf("%s", msg); // 印出該訊息
        for (int i=0; i<CMAX; i++) { // 廣播給其他人
            if (i != ci && clients[i].fd != 0) { // 如果對方不是發訊息者，而且不是空的，那就轉送給他！
                send(clients[i].fd, msg, strlen(msg)+1, 0);
            }
        }
    }
    close(cfd);
    clients[ci].fd = 0;
    return NULL;
}

void connectHandler(int sfd) {
    struct sockaddr_in raddr;
    socklen_t rAddrLen = sizeof(struct sockaddr);
    int cfd = accept(sfd, (struct sockaddr*) &raddr, &rAddrLen);
    for (int i=0; i<CMAX; i++) {
        if (clients[i].fd == 0) {
            memset(&clients[i], 0, sizeof(clients[i]));
            clients[i].fd = cfd;
            pthread_create(&clients[i].thread, NULL, receiver, &i);
            break;
        }
    }
}

int main(int argc, char *argv[]) {
    int port = atoi(argv[1]);
    printf("port=%d\n", port);
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in saddr, raddr;
    memset(&saddr, 0, sizeof(saddr));
    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(port);
    char msg[SMAX];
    saddr.sin_addr.s_addr = INADDR_ANY;
    int rb = bind(sfd, (struct sockaddr*) &saddr, sizeof(struct sockaddr));
    assert(rb >= 0);
    int rl = listen(sfd, CMAX);
    assert(rl >= 0);
    memset(clients, 0, sizeof(clients));
    while (1) {
        connectHandler(sfd);
    }
    close(sfd);
    return 0;
}
