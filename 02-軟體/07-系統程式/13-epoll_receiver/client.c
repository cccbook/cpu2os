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

void *receiver(void *argu) {
    int sfd = *(int*)argu;
    char msg[SMAX];
    while (1) {
        int n = recv(sfd, msg, SMAX, 0);
        if (n <=0) break;
        printf("%s", msg);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    char *name = argv[1];
    char *ip = argv[2];
    int port = atoi(argv[3]);
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in saddr, raddr;
    memset(&saddr, 0, sizeof(saddr));
    memset(&raddr, 0, sizeof(raddr));
    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(port);
    saddr.sin_addr.s_addr = inet_addr(ip);
    int rc = connect(sfd, (struct sockaddr*) &saddr, sizeof(struct sockaddr));
    assert(rc >= 0);
    pthread_t thread1;
    pthread_create(&thread1, NULL, receiver, &sfd);
    while (1) {
        char msg[SMAX], fmsg[SMAX];
        fgets(msg, SMAX, stdin);
        sprintf(fmsg, "%s: %s", name, msg);
        send(sfd, fmsg, strlen(fmsg)+1, 0);
    }
    close(sfd);
    return 0;
}
