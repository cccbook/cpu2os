#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>

#define SMAX 80

int main(int argc, char *argv[]) {
    int sfd = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in saddr, raddr;
    memset(&saddr, 0, sizeof(saddr));
    memset(&raddr, 0, sizeof(raddr));
    saddr.sin_family = AF_INET;
    int port = atoi(argv[1]);
    printf("port=%d\n", port);
    saddr.sin_port = htons(port);
    char msg[SMAX];
    if (argc<=2) { // server
        printf("I am server...\n");
        saddr.sin_addr.s_addr = INADDR_ANY;
        if (bind(sfd, (struct sockaddr*) &saddr, sizeof(struct sockaddr))!=0) {
            printf("error: bind fail! port %d occupied ...\n", port);
            exit(1);
        }
        socklen_t rAddrLen = sizeof(struct sockaddr);
        int rlen = recvfrom(sfd, msg, SMAX, 0, (struct sockaddr*) &raddr, &rAddrLen);
        printf("receive: %s from client addr %s\n", msg, inet_ntoa(raddr.sin_addr));
    } else { // client
        printf("I am client...\n");
        saddr.sin_addr.s_addr = inet_addr(argv[2]);
        memcpy(&raddr, &saddr, sizeof(saddr));
        char *connMsg = "<connect request>";
        sendto(sfd, connMsg, strlen(connMsg)+1, 0, (struct sockaddr*) &saddr, sizeof(struct sockaddr));
    }
    if (fork() == 0) {
        // child: receive message and print
        while (1) {
            socklen_t rAddrLen = sizeof(struct sockaddr);
            recvfrom(sfd, msg, SMAX, 0, (struct sockaddr*) &raddr, &rAddrLen);
            printf("receive: %s", msg);
        }
    } else {
        // parent: readline and send msg
        while (1) {
            fgets(msg, SMAX, stdin);
            sendto(sfd, msg, strlen(msg)+1, 0, (struct sockaddr*) &raddr, sizeof(struct sockaddr));
        }
    }
    close(sfd);
    return 0;
}
