#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#define KEYBOARD_PORT 0x60 // 鍵盤端口地址
#define KEY_ENTER 0x1C     // Enter 鍵的掃描碼
#define MOUSE_PORT 0x60    // 滑鼠端口地址
#define SERVER_IP "127.0.0.1" // 伺服器 IP 地址
#define SERVER_PORT 8888      // 伺服器端口

int main() {
    uint8_t key_code, mouse_data;
    int sockfd, ret;
    struct sockaddr_in server_addr;
    char sendbuf[1024] = "Hello, server!";
    char recvbuf[1024] = {0};

    // 建立 socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    // 設置伺服器地址
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(SERVER_PORT);

    // 連接伺服器
    ret = connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    if (ret < 0) {
        perror("connect");
        exit(EXIT_FAILURE);
    }

    // 輪詢鍵盤、滑鼠和網路
    while (1) {
        // 檢查鍵盤
        if ((key_code = inb(KEYBOARD_PORT)) == KEY_ENTER) {
            printf("Enter key pressed\n");
            break;
        }

        // 檢查滑鼠
        if ((mouse_data = inb(MOUSE_PORT)) != 0) {
            printf("Mouse data received: 0x%x\n", mouse_data);
            break;
        }

        // 檢查網路
        ret = recv(sockfd, recvbuf, sizeof(recvbuf), 0);
        if (ret > 0) {
            printf("Received: %s\n", recvbuf);
            break;
        }
    }

    close(sockfd);
    return 0;
}
