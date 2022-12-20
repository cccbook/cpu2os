# fs

## device file

* /dev/console : 主控台
    * 在視窗系統中，這是一個命令列視窗，在一般程式就是標準輸出入。
* /dev/tty : terminal, 鍵盤＋螢幕
* /dev/null : 像垃圾桶，寫進去的都會被捨棄。

## 標準呼叫與系統呼叫

標準呼叫會有緩衝 (buffered)，系統呼叫經常沒有緩衝

## proc

Linux 將所有東西都當成檔案，系統相關資訊都可以從 /proc 中取得。

```
guest@localhost:~$ ls /proc
1     16     24763  407  660  acpi         keys          softirqs
10    160    24767  413  661  buddyinfo    key-users     stat
108   161    24772  415  662  bus          kmsg          swaps
11    162    24773  417  663  cgroups      kpagecgroup   sys
1149  163    24890  418  664  cmdline      kpagecount    sysrq-trigger
1152  164    24891  420  665  consoles     kpageflags    sysvipc
1175  165    24899  430  678  cpuinfo      loadavg       thread-self
12    166    24907  469  691  crypto       locks         timer_list
13    17     25     519  692  devices      mdstat        tty
14    18     26     520  693  diskstats    meminfo       uptime
148   19     27     581  7    dma          misc          version
1488  2      275    592  709  driver       modules       version_signature
149   20     28     593  710  execdomains  mounts        vmallocinfo
15    21     283    594  77   fb           mtrr          vmstat
150   22     29     595  78   filesystems  net           zoneinfo
151   23     30     596  8    fs           pagetypeinfo
153   24     34     597  82   interrupts   partitions
154   24135  344    6    9    iomem        sched_debug
155   24357  345    602  91   ioports      schedstat
156   24513  35     604  923  irq          scsi
157   24514  4      625  934  kallsyms     self
159   24762  405    656  939  kcore        slabinfo

guest@localhost:~$ cat /proc/cpuinfo
processor       : 0
vendor_id       : GenuineIntel
cpu family      : 6
model           : 79
model name      : Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz
stepping        : 1
microcode       : 0x1
cpu MHz         : 2299.994
cache size      : 16384 KB
physical id     : 0
siblings        : 1
core id         : 0
cpu cores       : 1
apicid          : 0
initial apicid  : 0
fpu             : yes
fpu_exception   : yes
cpuid level     : 13
wp              : yes
flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology cpuid pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single pti ssbd ibrs ibpb fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm rdseed adx smap xsaveopt arat
bugs            : cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf
bogomips        : 4599.98
clflush size    : 64
cache_alignment : 64
address sizes   : 40 bits physical, 48 bits virtual
power management:

guest@localhost:~$ cat /proc/meminfo
MemTotal:        1009040 kB
MemFree:          116488 kB
MemAvailable:     334436 kB
Buffers:           72164 kB
Cached:           278388 kB
SwapCached:         1504 kB
Active:           436800 kB
Inactive:         364468 kB
Active(anon):     225604 kB
Inactive(anon):   245404 kB
Active(file):     211196 kB
Inactive(file):   119064 kB
Unevictable:        5408 kB
Mlocked:            5408 kB
SwapTotal:        262140 kB
SwapFree:         248060 kB
Dirty:                60 kB
Writeback:             0 kB
AnonPages:        455528 kB
Mapped:            96720 kB
Shmem:             17892 kB
Slab:              56056 kB
SReclaimable:      30416 kB
SUnreclaim:        25640 kB
KernelStack:        3084 kB
PageTables:         9100 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:      766660 kB
Committed_AS:     951872 kB
VmallocTotal:   34359738367 kB
VmallocUsed:           0 kB
VmallocChunk:          0 kB
HardwareCorrupted:     0 kB
AnonHugePages:         0 kB
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
CmaTotal:              0 kB
CmaFree:               0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
DirectMap4k:       67440 kB
DirectMap2M:      980992 kB
DirectMap1G:           0 kB
guest@localhost:~$ cat /proc/version
Linux version 4.15.0-32-generic (buildd@lgw01-amd64-039) (gcc version 7.3.0 (Ubuntu 7.3.0-16ubuntu3)) #35-Ubuntu SMP Fri Aug 10 17:58:07 UTC 2018

guest@localhost:~$ cat /proc/net/sockstat
sockets: used 185
TCP: inuse 8 orphan 1 tw 1 alloc 14 mem 2
UDP: inuse 1 mem 19
UDPLITE: inuse 0
RAW: inuse 0
FRAG: inuse 0 memory 0

guest@localhost:~$ cat /proc/sys/fs/file-max
94308
```
