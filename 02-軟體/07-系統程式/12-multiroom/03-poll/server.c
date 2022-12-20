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
#include <poll.h>

#define SMAX 80
#define CMAX 10

struct pollfd clients[CMAX], pfds[CMAX];

void *receiver(void *argu) {
    char msg[SMAX];
    while (1) {
        memcpy(pfds, clients, sizeof(clients));
        int timeout = 100;
        int ready = poll(pfds, CMAX, timeout);
        for (int ci = 0; ci < CMAX; ci++) {
            if (pfds[ci].revents != 0) { // 相較於 epoll，這裡還是要一個一個測。
                if (pfds[ci].revents & POLLIN) { // 接收到訊息的事件
                    ssize_t s = read(pfds[ci].fd, msg, sizeof(msg));
                    printf("%s", msg);
                    for (int i=0; i<CMAX; i++) { // 廣播給其他人
                        if (i != ci && clients[i].fd != 0) { // 如果對方不是發訊息者，而且不是空的，那就轉送給他！
                            write(clients[i].fd, msg, strlen(msg)+1);
                        }
                    }
                }
            }
        }
    }
    return NULL;
}

void connectHandler(int sfd) {
    struct sockaddr_in raddr;
    socklen_t rAddrLen = sizeof(struct sockaddr);
    int cfd = accept(sfd, (struct sockaddr*) &raddr, &rAddrLen); // 有客戶端連進來了，其串流為 cfd ...
    for (int i=0; i<CMAX; i++) { 
        if (clients[i].fd == 0) { // 找到一個空的客戶端
            memset(&clients[i], 0, sizeof(clients[i]));
            clients[i].events = POLLIN; // 監控其輸入
            clients[i].fd = cfd; // 監控對象為 cfd
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
    saddr.sin_port = htons(port); // htons(8888);
    char msg[SMAX];
    saddr.sin_addr.s_addr = INADDR_ANY;
    int rb = bind(sfd, (struct sockaddr*) &saddr, sizeof(struct sockaddr));
    assert(rb >= 0);
    int rl = listen(sfd, CMAX);
    assert(rl >= 0);
    memset(clients, 0, sizeof(clients));
    pthread_t thread1;
    pthread_create(&thread1, NULL, receiver, NULL);
    while (1) {
        connectHandler(sfd);
    }
    close(sfd);
    return 0;
}
