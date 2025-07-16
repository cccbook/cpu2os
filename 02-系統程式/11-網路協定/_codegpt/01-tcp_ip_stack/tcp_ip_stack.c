
/* tcp_ip_stack.c - Implementation of a simple TCP/IP stack */

#include "tcp_ip_stack.h"
#include <string.h>
#include <stdio.h>

/* Utility Functions */

uint16_t calculate_checksum(void *data, size_t length) {
    uint16_t *ptr = (uint16_t *)data;
    uint32_t sum = 0;

    for (size_t i = 0; i < length / 2; i++) {
        sum += ptr[i];
        if (sum > 0xFFFF) {
            sum -= 0xFFFF;
        }
    }

    if (length % 2 == 1) {
        sum += ((uint8_t *)data)[length - 1] << 8;
        if (sum > 0xFFFF) {
            sum -= 0xFFFF;
        }
    }

    return ~((uint16_t)sum);
}

/* Network Layer Functions */

void send_ipv4_packet(IPv4Header *header, uint8_t *payload, size_t payload_length) {
    header->total_length = htons(sizeof(IPv4Header) + payload_length);
    header->checksum = 0;
    header->checksum = calculate_checksum(header, sizeof(IPv4Header));

    // Add logic to send the packet over a raw socket (to be implemented).
}

void receive_ipv4_packet(uint8_t *packet, size_t length) {
    IPv4Header *header = (IPv4Header *)packet;

    // Verify checksum
    uint16_t received_checksum = header->checksum;
    header->checksum = 0;
    if (calculate_checksum(header, sizeof(IPv4Header)) != received_checksum) {
        printf("Invalid checksum\n");
        return;
    }

    // Process payload based on the protocol
    uint8_t *payload = packet + sizeof(IPv4Header);
    size_t payload_length = length - sizeof(IPv4Header);

    switch (header->protocol) {
        case PROTOCOL_UDP:
            receive_udp_packet(header, payload, payload_length);
            break;
        case PROTOCOL_TCP:
            receive_tcp_packet(header, payload, payload_length);
            break;
        default:
            printf("Unknown protocol: %d\n", header->protocol);
    }
}

/* Transport Layer Functions */

void send_udp_packet(uint32_t source_ip, uint16_t source_port,
                     uint32_t dest_ip, uint16_t dest_port,
                     uint8_t *data, size_t length) {
    UDPHeader udp_header;
    udp_header.source_port = htons(source_port);
    udp_header.dest_port = htons(dest_port);
    udp_header.length = htons(sizeof(UDPHeader) + length);
    udp_header.checksum = 0;

    // Create IPv4 header
    IPv4Header ip_header;
    ip_header.version_ihl = (IP_VERSION << 4) | (sizeof(IPv4Header) / 4);
    ip_header.tos = 0;
    ip_header.id = htons(0);
    ip_header.flags_offset = htons(0);
    ip_header.ttl = DEFAULT_TTL;
    ip_header.protocol = PROTOCOL_UDP;
    ip_header.source_ip = htonl(source_ip);
    ip_header.dest_ip = htonl(dest_ip);

    // Add logic to send the packet (to be implemented).
}

void receive_udp_packet(IPv4Header *ip_header, uint8_t *data, size_t length) {
    if (length < sizeof(UDPHeader)) {
        printf("Incomplete UDP packet\n");
        return;
    }

    UDPHeader *udp_header = (UDPHeader *)data;
    uint8_t *payload = data + sizeof(UDPHeader);
    size_t payload_length = length - sizeof(UDPHeader);

    printf("Received UDP packet: Source Port %u, Destination Port %u\n",
           ntohs(udp_header->source_port), ntohs(udp_header->dest_port));
}

void send_tcp_packet(uint32_t source_ip, uint16_t source_port,
                     uint32_t dest_ip, uint16_t dest_port,
                     uint8_t *data, size_t length, uint8_t flags) {
    TCPHeader tcp_header;
    memset(&tcp_header, 0, sizeof(TCPHeader));
    tcp_header.source_port = htons(source_port);
    tcp_header.dest_port = htons(dest_port);
    tcp_header.flags = flags;
    tcp_header.checksum = 0;

    // Create IPv4 header
    IPv4Header ip_header;
    ip_header.version_ihl = (IP_VERSION << 4) | (sizeof(IPv4Header) / 4);
    ip_header.tos = 0;
    ip_header.id = htons(0);
    ip_header.flags_offset = htons(0);
    ip_header.ttl = DEFAULT_TTL;
    ip_header.protocol = PROTOCOL_TCP;
    ip_header.source_ip = htonl(source_ip);
    ip_header.dest_ip = htonl(dest_ip);

    // Add logic to send the packet (to be implemented).
}

void receive_tcp_packet(IPv4Header *ip_header, uint8_t *data, size_t length) {
    if (length < sizeof(TCPHeader)) {
        printf("Incomplete TCP packet\n");
        return;
    }

    TCPHeader *tcp_header = (TCPHeader *)data;
    uint8_t *payload = data + sizeof(TCPHeader);
    size_t payload_length = length - sizeof(TCPHeader);

    printf("Received TCP packet: Source Port %u, Destination Port %u\n",
           ntohs(tcp_header->source_port), ntohs(tcp_header->dest_port));
}
