

* ChatGPT: PCap 如何捕獲封包
    * https://chatgpt.com/c/671b2e43-9fd0-8012-b556-fa78d0f21bc5

## ccc: libpcap 在 linux 上是如何捕獲網路封包的

`libpcap` 在 Linux 上捕獲網路封包的過程依賴於操作系統內核提供的「封包捕獲機制」，這個機制允許應用程式直接存取網路介面上的資料封包。其運作方式可以分為以下幾個步驟：

### 1. **網路介面設置為混雜模式（Promiscuous Mode）**
在一般情況下，網路介面卡只會接收發給它自己的資料封包（即目標 MAC 位址是自己的封包）。然而，為了能夠捕獲網路上所有經過的封包，`libpcap` 通常會將網路介面設置為混雜模式。這樣網卡就可以接收網路上所有的封包，而不只是發給它自己的那些封包。

混雜模式下，網卡能接收到屬於其他設備的封包，從而可以進行全網段的封包捕獲。

### 2. **使用 Linux 內核中的「套接字」機制**
`libpcap` 主要是通過原始套接字（raw socket）來捕獲封包。Linux 提供了一種特殊的套接字類型，稱為原始套接字，它允許應用程式接收和發送數據鏈路層（OSI 模型第二層）的封包。這意味著 `libpcap` 可以在資料進入 TCP/IP 協議棧之前就捕獲封包，這些封包包含了鏈路層的標頭（例如 Ethernet 標頭）。

此外，為了提高效能，`libpcap` 還可能會使用一種名為 **PF_PACKET** 的套接字。這是一種 Linux 特有的套接字類型，能夠比傳統的原始套接字更高效地捕獲網路封包。

### 3. **內核中的封包抓取**
當網路介面卡接收到封包後，這些封包會被傳送到網路驅動程式，再交給 Linux 內核處理。此時，`libpcap` 透過前面提到的原始套接字或 PF_PACKET 套接字從內核中接收這些封包。內核會把捕獲的封包複製到一個記憶體緩衝區，然後將其交給 `libpcap`。

### 4. **封包過濾（使用 BPF）**
為了減少不必要的資料處理和提高效率，`libpcap` 可以使用一種稱為 **BPF（Berkeley Packet Filter）** 的過濾機制。這允許使用者定義一個過濾條件，只捕獲特定類型的封包，而忽略其他無關的封包。BPF 過濾器在內核層運行，因此能夠有效地減少應用層接收到的封包數量，降低了 CPU 負擔。

例如，若只對 TCP 封包感興趣，可以通過 BPF 設定過濾條件，讓內核只傳送符合條件的封包給 `libpcap`。

### 5. **用戶空間處理**
最後，封包通過內核傳遞給 `libpcap`，並被複製到用戶空間的緩衝區中。應用程式可以使用 `libpcap` 提供的 API 來讀取這些封包，進行進一步的分析或處理。

### 6. **抓取封包的生命周期**
當抓取結束後，`libpcap` 會關閉相關的套接字，並將網路介面從混雜模式中退出，恢復正常運作模式。

### 小結
總體來說，`libpcap` 在 Linux 上依賴於內核提供的網路封包捕獲功能，主要是通過套接字（尤其是 PF_PACKET 套接字）與內核進行封包的交換，並且使用 BPF 來進行封包過濾。這樣的結構確保了封包捕獲的效率和靈活性。

## ccc: raw socket 如何運作

原始套接字（Raw Socket）是一種特殊的網路套接字，允許應用程式直接讀取或寫入網路層（OSI 模型的第 3 層）甚至資料鏈路層（OSI 模型的第 2 層）的資料封包。這些套接字通常由需要對網路封包進行細粒度控制的應用程式使用，例如網路封包分析器（如 `tcpdump`）、網路掃描器或自定義協議的實作。

在 Linux 上，原始套接字的運作過程可分為以下幾個主要步驟：

### 1. **套接字創建**
開發人員使用 `socket()` 系統呼叫，並將 `AF_INET` 或 `AF_PACKET` 與 `SOCK_RAW` 一起傳入，以創建原始套接字。以下是兩種創建方式的區別：
   - **AF_INET**：此類型的原始套接字允許操作網路層（例如 IP 層）的封包。可以捕獲或傳送的封包通常包括 IP 標頭及其後的資料。
   - **AF_PACKET**：這是 Linux 特有的功能，允許操作資料鏈路層的封包。這類套接字可以存取完整的封包，包括 Ethernet 標頭，適合需要獲取鏈路層資訊的應用程式。

```c
// 創建 AF_INET 原始套接字
int sock = socket(AF_INET, SOCK_RAW, IPPROTO_TCP);

// 創建 AF_PACKET 原始套接字
int sock = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
```

### 2. **設置權限**
原始套接字能夠讀寫底層封包，屬於高權限操作。為了防止安全問題，只有 `root` 或具有特權的使用者可以創建和使用原始套接字。如果一般使用者想要使用，通常需要授予應用程式 `CAP_NET_RAW` 權限。

### 3. **接收封包**
通過原始套接字接收封包時，網卡會將接收到的封包傳遞給網路協議棧，然後將封包複製到應用程式的緩衝區中。應用程式可以使用 `recvfrom()` 或 `recvmsg()` 來接收封包。

對於 AF_INET 原始套接字，應用程式可以直接讀取 IP 標頭和後續的協議數據。例如，如果套接字設置為接收 `IPPROTO_TCP`，則只會接收 TCP 封包，而非其他協議的封包。

對於 AF_PACKET 原始套接字，接收的封包包括資料鏈路層（例如 Ethernet）標頭以及後續數據。這樣能夠獲得更多的底層資訊，並進行更精細的分析。

### 4. **封包構建與傳送**
原始套接字不僅能夠接收封包，還允許應用程式構建並傳送自定義封包。應用程式需要自行構建協議標頭（例如 Ethernet、IP、TCP 標頭等）並填充有效負載。這樣的封包構建對進行協議測試、自定義協議實作等情況很有用。

以下是一個簡單的傳送封包範例：

```c
// 創建 IP 標頭 + 資料的緩衝區
unsigned char buffer[1024];
struct iphdr *iph = (struct iphdr *) buffer;

// 填充 IP 標頭
iph->version = 4; // IPv4
iph->ihl = 5;
iph->tot_len = htons(40); // 假設封包長度為 40 字節
iph->protocol = IPPROTO_TCP; // TCP 協議
// ... 填充其他欄位

// 使用 sendto 傳送封包
sendto(sock, buffer, iph->tot_len, 0, (struct sockaddr *)&dest, sizeof(dest));
```

### 5. **進行封包過濾**
為了減少不必要的封包處理，原始套接字可以搭配 BPF（Berkeley Packet Filter）進行過濾。這樣可以在內核中指定只接收特定類型的封包，提高效率。

### 6. **接收與傳送過程中的封包校驗**
原始套接字的特性之一是允許應用程式完全控制封包的標頭。因此，校驗和的計算也成為了應用程式的責任。特別是 IP 層或傳輸層（如 TCP 或 UDP）的校驗和需要正確計算，否則封包可能被網路設備或目標設備丟棄。

### 小結
在 Linux 上，原始套接字允許應用程式存取和操作網路層甚至鏈路層的封包，提供了非常靈活的封包操作能力。這種靈活性對網路分析、安全檢測、自定義協議實作等應用場景非常重要，但也因其高權限需求而需要注意安全性。

## ccc: 如何用程式將網路介面卡設置為混雜模式

在 Linux 上，可以使用 `ioctl` 系統呼叫來將網路介面卡設置為混雜模式。以下是使用 C 語言程式設置混雜模式的範例：

```c
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <sys/socket.h>

int main() {
    int sockfd;
    struct ifreq ifr;

    // 建立一個套接字
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return 1;
    }

    // 指定網路介面，例如 "eth0"
    strncpy(ifr.ifr_name, "eth0", IFNAMSIZ);

    // 取得目前的介面標誌
    if (ioctl(sockfd, SIOCGIFFLAGS, &ifr) == -1) {
        perror("SIOCGIFFLAGS");
        close(sockfd);
        return 1;
    }

    // 設置混雜模式標誌
    ifr.ifr_flags |= IFF_PROMISC;

    // 更新介面標誌以啟用混雜模式
    if (ioctl(sockfd, SIOCSIFFLAGS, &ifr) == -1) {
        perror("SIOCSIFFLAGS");
        close(sockfd);
        return 1;
    }

    printf("混雜模式已啟用。\n");

    // 記得最後清理套接字
    close(sockfd);
    return 0;
}
```

### 說明

1. **建立套接字**：這裡使用 `socket(AF_INET, SOCK_DGRAM, 0)` 建立一個套接字，用來進行 `ioctl` 呼叫。
2. **設置介面名稱**：用 `strncpy` 將目標網路介面名稱（例如 `eth0`）複製到 `ifr.ifr_name` 中。
3. **取得和設定標誌**：
   - 使用 `SIOCGIFFLAGS` 取得目前的介面標誌。
   - 修改 `ifr_flags` 欄位，設置 `IFF_PROMISC` 以啟用混雜模式。
4. **更新設定**：使用 `SIOCSIFFLAGS` 更新網路介面的標誌，將設定套用到指定的介面。
5. **關閉套接字**：結束後記得關閉套接字。

這樣，`eth0` 介面就會被設置為混雜模式。
