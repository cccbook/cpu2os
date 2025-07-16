/* tcp_ip_stack.h - Header file for a simple TCP/IP stack */

#ifndef TCP_IP_STACK_H
#define TCP_IP_STACK_H

#include <stdint.h>
#include <stddef.h>

/* Constants */
#define MAX_PACKET_SIZE 1500 // Maximum Ethernet frame payload size
#define IP_VERSION 4         // IPv4
#define DEFAULT_TTL 64       // Default Time-to-Live

/* IP Protocol Numbers */
#define PROTOCOL_ICMP 1
#define PROTOCOL_TCP  6
#define PROTOCOL_UDP  17

/* Structure Definitions */

/* Ethernet Frame */
typedef struct {
    uint8_t destination[6];
    uint8_t source[6];
    uint16_t ethertype;
    uint8_t payload[MAX_PACKET_SIZE];
} EthernetFrame;

/* IPv4 Header */
typedef struct {
    uint8_t version_ihl;     // Version and Internet Header Length
    uint8_t tos;             // Type of Service
    uint16_t total_length;   // Total Length
    uint16_t id;             // Identification
    uint16_t flags_offset;   // Flags and Fragment Offset
    uint8_t ttl;             // Time-to-Live
    uint8_t protocol;        // Protocol
    uint16_t checksum;       // Header Checksum
    uint32_t source_ip;      // Source IP Address
    uint32_t dest_ip;        // Destination IP Address
} IPv4Header;

/* UDP Header */
typedef struct {
    uint16_t source_port;    // Source Port
    uint16_t dest_port;      // Destination Port
    uint16_t length;         // Length
    uint16_t checksum;       // Checksum
} UDPHeader;

/* TCP Header */
typedef struct {
    uint16_t source_port;    // Source Port
    uint16_t dest_port;      // Destination Port
    uint32_t sequence;       // Sequence Number
    uint32_t acknowledgment; // Acknowledgment Number
    uint8_t data_offset;     // Data Offset and Reserved Bits
    uint8_t flags;           // Flags
    uint16_t window;         // Window Size
    uint16_t checksum;       // Checksum
    uint16_t urgent_pointer; // Urgent Pointer
} TCPHeader;

/* Function Prototypes */

/* Utility Functions */
uint16_t calculate_checksum(void *data, size_t length);

/* Network Layer Functions */
void send_ipv4_packet(IPv4Header *header, uint8_t *payload, size_t payload_length);
void receive_ipv4_packet(uint8_t *packet, size_t length);

/* Transport Layer Functions */
void send_udp_packet(uint32_t source_ip, uint16_t source_port,
                     uint32_t dest_ip, uint16_t dest_port,
                     uint8_t *data, size_t length);
void receive_udp_packet(IPv4Header *ip_header, uint8_t *data, size_t length);

void send_tcp_packet(uint32_t source_ip, uint16_t source_port,
                     uint32_t dest_ip, uint16_t dest_port,
                     uint8_t *data, size_t length, uint8_t flags);
void receive_tcp_packet(IPv4Header *ip_header, uint8_t *data, size_t length);

#endif /* TCP_IP_STACK_H */
