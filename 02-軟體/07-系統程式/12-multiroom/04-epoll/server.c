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
#include <sys/epoll.h>

#define SMAX 80
#define CMAX 10

struct Client_t {
    int fd;
};

struct Client_t clients[CMAX];

int epollfd;

void *receiver(void *argu) {
    char msg[SMAX];
    while (1) {
        int timeout = 100;
        struct epoll_event events[CMAX];
        int event_count = epoll_wait(epollfd, events, CMAX, timeout); // 等待事件發生，event_count 是事件數
        for(int i = 0; i < event_count; i++) { // 這裡比 poll 好，因為只要測有事件發生的串流，而不需要全測！
            int n = read(events[i].data.fd, msg, SMAX);
            msg[n] = '\0';
            printf("%s", msg);
            for (int ci=0; ci<CMAX; ci++) { // broadcast
                if (clients[ci].fd != 0 && clients[ci].fd != events[i].data.fd) {
                    write(clients[ci].fd, msg, strlen(msg)+1);
                }
            }
        }
    }
    return NULL;
}

void connectHandler(int sfd) {
    struct sockaddr_in raddr;
    socklen_t rAddrLen = sizeof(struct sockaddr);
    int cfd = accept(sfd, (struct sockaddr*) &raddr, &rAddrLen);
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = cfd;
    epoll_ctl(epollfd, EPOLL_CTL_ADD, cfd, &ev);
    for (int i=0; i<CMAX; i++) { // 新增 cfd 到 clients
        if (clients[i].fd == 0) {
            clients[i].fd = cfd;
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
    epollfd = epoll_create1(0);
    pthread_t thread1;
    pthread_create(&thread1, NULL, receiver, NULL);
    while (1) {
        connectHandler(sfd);
    }
    close(sfd);
    return 0;
}
