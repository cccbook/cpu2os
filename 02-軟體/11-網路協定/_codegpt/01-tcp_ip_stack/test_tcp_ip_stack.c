#include "tcp_ip_stack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h> // for htons, htonl

int main() {
    // 測試 IPv4 封包發送
    uint8_t payload[] = "Hello, world!";
    IPv4Header ip_header;
    memset(&ip_header, 0, sizeof(ip_header));

    ip_header.version_ihl = (IP_VERSION << 4) | (sizeof(IPv4Header) / 4);
    ip_header.ttl = DEFAULT_TTL;
    ip_header.protocol = PROTOCOL_UDP;
    ip_header.source_ip = htonl(0x7F000001); // 127.0.0.1
    ip_header.dest_ip = htonl(0x7F000001);   // 127.0.0.1

    printf("Sending IPv4 packet...\n");
    send_ipv4_packet(&ip_header, payload, sizeof(payload) - 1);

    // 測試 UDP 封包發送
    printf("Sending UDP packet...\n");
    send_udp_packet(0x7F000001, 12345, 0x7F000001, 54321, payload, sizeof(payload) - 1);

    // 測試 TCP 封包發送
    printf("Sending TCP packet...\n");
    send_tcp_packet(0x7F000001, 12345, 0x7F000001, 54321, payload, sizeof(payload) - 1, 0x18); // PSH + ACK

    return 0;
}
