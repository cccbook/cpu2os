# 第八章。後面的過程


## 審查可執行文件

我們知道在內存中運行的程序在代碼(由於歷史原因也稱為文本)和數據中有兩個主要組件。然而，我們也知道，一個可執行文件的生命周期並不在內存中，而是在等待加載運行的磁盤上以文件的形式度過。由於文件本質上只是一個連續的位數組，所以所有系統都提出了在文件中組織代碼和數據以按需執行的方法。這種文件格式通常稱為二進制文件或可執行文件。文件的位和字節通常以一種可直接由處理器硬件解釋的格式存儲在內存中。


## 表示可執行文件


### 三個標准段

至少，任何可執行文件格式都需要指定二進制文件中代碼和數據的位置。這是可執行文件中的兩個主要段。


我們到目前為止還沒有提到的另一個組件是未初始化全局變量的存儲空間。如果我們聲明一個變量並給它一個初始值，這個值需要存儲在可執行文件中，以便在程序啟動時它可以初始化到正確的值。然而，當程序首次執行時，許多變量是未初始化的(或零)。在可執行文件中為這些文件騰出空間，然後簡單地存儲0或NULL值是浪費空間，不必要地在磁盤上膨脹可執行文件大小。因此，大多數二進制格式都將額外的BSS段定義為零初始化數據的占位符大小。在程序中加載BSS描述的額外內存可以被分配(並設置為零!)BSS可能表示由Symbol(舊IBM計算機的彙編命令)啟動的塊;確切的推導過程可能已經被遺忘了。


### 二進制格式

可執行文件是由源代碼中的工具鏈創建的。這個文件需要顯式定義的格式，以便編譯器可以創建它，操作系統可以識別它並加載到內存中，將它轉換為操作系統可以管理的正在運行的進程。這種可執行文件格式可以特定於操作系統，因為我們通常不會期望為一個系統編譯的程序會在另一個系統上執行(例如，您不希望您的Windows程序在Linux上運行，或者您的Linux程序在OS X上運行)。


然而，所有可執行文件格式之間的共同線程是，它們包含一個預定義的、標准化的頭文件，描述程序代碼和數據如何存儲在文件的其余段。換句話說，它通常描述為“程序代碼從20字節開始進入這個文件，長度為50千字節”。程序數據跟在它後面，長度為20千字節。


最近，一種特定的格式已經成為現代UNIX類型系統可執行表示的實際標准。它被稱為可執行和鏈接器格式，簡稱ELF;我們很快會更詳細地研究它。

### 二進制格式歷史


#### a.out

ELF並不總是標准;最初的UNIX系統使用一種稱為a.out的文件格式。如果您編譯一個沒有-o選項來指定輸出文件名的程序，我們可以看到這一點的痕跡;可執行文件將以默認名稱a.out創建 [23]。


一個 .out 是一種非常簡單的頭文件格式，只允許一個數據、代碼和 BSS 段。正如您將看到的，這對於具有動態庫的現代系統來說是不夠的。


#### COFF


普通對像文件格式(COFF)是ELF的前身。它的頭格式更加靈活，允許文件中有更多(但有限)段。


COFF對共享庫的優雅支持也有困難，因此選擇ELF作為Linux上的替代實現。


然而，COFF作為可移植的可執行文件或PE格式存在於Microsoft Windows中。PE對於Windows就像ELF對於Linux一樣。


實際上是>[23]。輸出是鏈接器的默認輸出文件名。編譯器通常使用隨機生成的文件名作為彙編和目標代碼的中間文件。

## ELF



ELF是一種在系統中表示二進制代碼的非常靈活的格式。通過遵循ELF標准，您可以像普通可執行程序或系統庫一樣輕松地表示內核二進制文件。同樣的工具可以用於檢查和操作所有ELF文件，理解ELF文件格式的開發人員可以將他們的技能應用到大多數現代UNIX系統中。


ELF擴展了COFF，並提供了足夠的靈活性來定義任意數量的分段，每個分段都有自己的屬性。這有助於更容易的動態鏈接和調試。


> 圖8.1。ELF的概述


![](http://www.bottomupcs.com/chapter07/figures/elf-overview.png)


> ELF概述



### ELF文件頭


總的來說，文件有一個文件頭，通常描述文件，然後有指向組成文件的各個段的指針。例8.1“ELF頭”顯示了ELF32(32位ELF形式)API文檔中給出的描述。這是定義ELF頭的C結構的布局。


> 8.1的例子。ELF頭
```
typedef struct {
        unsigned char e_ident[EI_NIDENT];
        Elf32_Half    e_type;
        Elf32_Half    e_machine;
        Elf32_Word    e_version;
        Elf32_Addr    e_entry;
        Elf32_Off     e_phoff;
        Elf32_Off     e_shoff;
        Elf32_Word    e_flags;
        Elf32_Half    e_ehsize;
        Elf32_Half    e_phentsize;
        Elf32_Half    e_phnum;
        Elf32_Half    e_shentsize;
        Elf32_Half    e_shnum;
        Elf32_Half    e_shstrndx;
} Elf32_Ehdr;
```

> Example 8.2. The ELF Header, as shown by readelf 

```
$ readelf --header /bin/ls

ELF Header:
  Magic:   7f 45 4c 46 01 02 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF32
  Data:                              2's complement, big endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              EXEC (Executable file)
  Machine:                           PowerPC
  Version:                           0x1
  Entry point address:               0x10002640
  Start of program headers:          52 (bytes into file)
  Start of section headers:          87460 (bytes into file)
  Flags:                             0x0
  Size of this header:               52 (bytes)
  Size of program headers:           32 (bytes)
  Number of program headers:         8
  Size of section headers:           40 (bytes)
  Number of section headers:         29
  Section header string table index: 28

  [...]
```

例8.2 “ELF報頭，如readelf所示” 顯示一個人類可讀的表單，如readelf程序所示，該程序是GNU binutils的一部分。


e_ident數組是任何ELF文件開頭的第一個元素，並且總是以一些“神奇的”字節開始。第一個字節是0x7F，接下來的三個字節是“ELF”。您可以通過使用hexdump命令檢查ELF二進制文件來親自查看。


> 8.3的例子。檢查小ELF的魔法號碼

```
  1 ianw@mingus:~$ hexdump -C /bin/ls | more
  2 00000000  7f 45 4c 46 01 02 01 00  00 00 00 00 00 00 00 00  |.ELF............|
  3 
  4 ... (rest of the program follows) ...
````

注意開始是0x7F，然後是ASCII編碼的“ELF”字符串。看看這個標准，看看數組的其余部分定義了什麼，以及二進制中的值是什麼。


接下來，我們為這個二進制文件創建的機器類型設置了一些標志。首先我們可以看到ELF定義了不同類型大小的版本，一個為32位版本，一個為64位版本;這裡我們檢查32位版本。區別主要在於64位機器上的地址顯然需要保存在64位變量中。我們可以看到，二進制已經為一個使用2的補碼來表示負數的大型endian機器創建好了。略過一點，我們可以看到機器告訴我們這是一個PowerPC二進制文件。


看似無害的入口地址似乎足夠直接;這是程序代碼開始時內存中的地址。最初的C程序員被告知main()是程序中調用的第一個程序。使用入口點地址，我們可以驗證它不是。


> 8.4的例子。調查入口點

```
$ cat test.c
#include <stdio.h>

int main(void)
{
        printf("main is : %p\n", &main);
        return 0;
}

$ gcc -Wall -o test test.c

$ ./test
main is : 0x10000430

$ readelf --headers ./test | grep 'Entry point'
  Entry point address:               0x100002b0

$ objdump --disassemble ./test | grep 100002b0
100002b0 <_start>:
100002b0:       7c 29 0b 78     mr      r9,r1
```

在示例8.4中，“調查入口點”我們可以看到入口點實際上是一個名為_start的函數。我們的程序根本沒有定義這個，最重要的下劃線表明它在一個單獨的命名空間中。我們將在“啟動程序”一節中詳細分析程序是如何啟動的。


之後，頭文件包含指向文件中ELF文件開始的其他重要部分的指針，比如目錄表。


### 符號和重定位


ELF規範提供了符號表，它只是將字符串(符號)映射到文件中的位置。連接需要符號;例如，為聲明為extern int foo的變量賦值;將要求鏈接器查找foo的地址，這將涉及在符號表中查找“foo”並查找地址。


與符號密切相關的是重定位。重新安置只是留下一個空白的空間，以後再補上。在前面的示例中，在知道foo的地址之前，不能使用它。然而,在32位系統上,我們知道foo的地址必須是一個4字節的值,所以任何時候編譯器需要使用這個地址(說,分配一個值),它可以簡單地離開4個字節的空白空間,保持搬遷,本質上說鏈接器”的真正價值”foo”到4字節的地址”。如前所述，這需要解析符號“foo”。“重定位”段包含關於重定位的更多信息。


### Section 段和 Segment 段


ELF格式指定了一個ELF文件的兩個“視圖”——一個用於鏈接，另一個用於執行。這為系統設計人員提供了很大的靈活性。


我們將討論目標代碼中等待鏈接到可執行文件的部分。一個或多個部分映射到可執行文件中的一個段。


### Segments 段

正如我們以前所做的那樣，在檢查較低的層之前，有時更容易查看較高層次的抽像(片段)。


如前所述，ELF文件有一個描述文件總體布局的頭文件。ELF報頭實際上指向另一組名為程序報頭的報頭。這些標頭向操作系統描述了它將二進制文件加載到內存並執行所需的任何內容。segments 段是由程序頭描述的，但是讓可執行文件運行還需要一些其他的東西。


> 8.5的例子。程序標題

```
typedef struct {
          Elf32_Word p_type;
          Elf32_Off  p_offset;
          Elf32_Addr p_vaddr;
          Elf32_Addr p_paddr;
          Elf32_Word p_filesz;
          Elf32_Word p_memsz;
          Elf32_Word p_flags;
          Elf32_Word p_align;
}
```


程序頭的定義見例8.5“程序頭”。你可能已經從上面的ELF頭定義中注意到了字段e_phoff, e_phnum和e_phentsize的存在;這些只是程序頭開始的文件中的偏移量，有多少程序頭，每個程序頭有多大。有了這3位信息，您可以很容易地找到和讀取程序頭。


程序頭不僅僅是段。p_type字段只定義程序頭定義的內容。例如，如果這個字段是PT_INTERP，則頭文件定義為指向二進制文件解釋器的字符串指針。我們之前討論過編譯語言和解釋語言，並指出編譯器構建的二進制代碼可以獨立運行。為什麼需要一個解釋器?和往常一樣，真實的情況要復雜一些。現代系統在加載可執行文件時需要靈活性有幾個原因，要做到這一點，一些信息只能在程序設置為運行時獲得。在以後的章節中，我們會看到動態鏈接。因此，可能需要對二進制文件進行一些小的修改，以使其在運行時正常工作。因此，二進制文件的通常解釋器是動態加載器，之所以這麼叫它是因為它需要完成可執行文件的加載並准備二進制映像以運行。


在p_type字段中使用PT_LOAD值來描述段。然後，程序頭中的其他字段描述每個段。p_offset字段告訴您段數據到磁盤上文件的深度。p_vaddr字段告訴您數據駐留在虛擬內存中的地址(p_addr描述物理地址，這只對不實現虛擬內存的小型嵌入式系統非常有用)。兩個標志p_filesz和p_memsz用於告訴您磁盤上的段有多大以及內存中應該有多大。如果內存大小大於磁盤大小，那麼重疊部分應該用零填充。通過這種方式，不必為空的全局變量浪費空間，就可以在二進制文件中節省大量空間。最後，p_flags表示段上的權限。可任意組合指定執行、讀寫權限;例如，代碼段應該標記為只讀和執行，數據段應該標記為讀和寫，而不是執行。


在程序標頭中定義了一些其他的段類型，它們在標准規範中有更完整的描述。


###  段 Section

正如我們所提到的，分段構成分段。section是將二進制文件組織到邏輯區域的一種方法，用於在編譯器和鏈接器之間通信信息。在一些特殊的二進制文件中，例如Linux內核，段以更特定的方式使用(請參閱“定制段”一節)。


我們已經看到了段最終如何歸結為磁盤上文件中的數據團，並描述了應該在哪裡加載它以及它擁有哪些權限。section具有類似於segment的頭信息，如例8.6所示，“Sections”。


> 8.6的例子。 Section 段

```c
typedef struct {
          Elf32_Word sh_name;
          Elf32_Word sh_type;
          Elf32_Word sh_flags;
          Elf32_Addr sh_addr;
          Elf32_Off  sh_offset;
          Elf32_Word sh_size;
          Elf32_Word sh_link;
          Elf32_Word sh_info;
          Elf32_Word sh_addralign;
          Elf32_Word sh_entsize;
}
```

段還有一些為sh_type字段定義的類型;例如，SH_PROGBITS類型的段定義為保存二進制數據供程序使用的部分。其他標志表示，如果這個段是一個符號表(例如鏈接器或調試器使用)，或者是動態加載器使用的東西。還有更多的屬性，例如這個段需要為它分配內存的allocation屬性。


下面我們將檢查示例8.7中列出的程序，“節”。


> 8.7的例子。Section 段

```c
#include <stdio.h>

int big_big_array[10*1024*1024];

char *a_string = "Hello, World!";

int a_var_with_value = 0x100;

int main(void)
{
	big_big_array[0] = 100;
	printf("%s\n", a_string);
	a_var_with_value += 20;
}
```


例8.8，“section readelf output”顯示了readelf的輸出，其中一些段的清晰度被剝離。使用這個輸出，我們可以分析我們的簡單程序的每個段，並看到它在最終的輸出二進制文件中結束。


> 8.8的例子。Section 段 readelf輸出

```
$ readelf --all ./sections
ELF Header:
 ...
  Size of section headers:           40 (bytes)
  Number of section headers:         37
  Section header string table index: 34

Section Headers:
  [Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            00000000 000000 000000 00      0   0  0
  [ 1] .interp           PROGBITS        10000114 000114 00000d 00   A  0   0  1
  [ 2] .note.ABI-tag     NOTE            10000124 000124 000020 00   A  0   0  4
  [ 3] .hash             HASH            10000144 000144 00002c 04   A  4   0  4
  [ 4] .dynsym           DYNSYM          10000170 000170 000060 10   A  5   1  4
  [ 5] .dynstr           STRTAB          100001d0 0001d0 00005e 00   A  0   0  1
  [ 6] .gnu.version      VERSYM          1000022e 00022e 00000c 02   A  4   0  2
  [ 7] .gnu.version_r    VERNEED         1000023c 00023c 000020 00   A  5   1  4
  [ 8] .rela.dyn         RELA            1000025c 00025c 00000c 0c   A  4   0  4
  [ 9] .rela.plt         RELA            10000268 000268 000018 0c   A  4  25  4
  [10] .init             PROGBITS        10000280 000280 000028 00  AX  0   0  4
  [11] .text             PROGBITS        100002b0 0002b0 000560 00  AX  0   0 16
  [12] .fini             PROGBITS        10000810 000810 000020 00  AX  0   0  4
  [13] .rodata           PROGBITS        10000830 000830 000024 00   A  0   0  4
  [14] .sdata2           PROGBITS        10000854 000854 000000 00   A  0   0  4
  [15] .eh_frame         PROGBITS        10000854 000854 000004 00   A  0   0  4
  [16] .ctors            PROGBITS        10010858 000858 000008 00  WA  0   0  4
  [17] .dtors            PROGBITS        10010860 000860 000008 00  WA  0   0  4
  [18] .jcr              PROGBITS        10010868 000868 000004 00  WA  0   0  4
  [19] .got2             PROGBITS        1001086c 00086c 000010 00  WA  0   0  1
  [20] .dynamic          DYNAMIC         1001087c 00087c 0000c8 08  WA  5   0  4
  [21] .data             PROGBITS        10010944 000944 000008 00  WA  0   0  4
  [22] .got              PROGBITS        1001094c 00094c 000014 04 WAX  0   0  4
  [23] .sdata            PROGBITS        10010960 000960 000008 00  WA  0   0  4
  [24] .sbss             NOBITS          10010968 000968 000000 00  WA  0   0  1
  [25] .plt              NOBITS          10010968 000968 000060 00 WAX  0   0  4
  [26] .bss              NOBITS          100109c8 000968 2800004 00  WA  0   0  4
  [27] .comment          PROGBITS        00000000 000968 00018f 00      0   0  1
  [28] .debug_aranges    PROGBITS        00000000 000af8 000078 00      0   0  8
  [29] .debug_pubnames   PROGBITS        00000000 000b70 000025 00      0   0  1
  [30] .debug_info       PROGBITS        00000000 000b95 0002e5 00      0   0  1
  [31] .debug_abbrev     PROGBITS        00000000 000e7a 000076 00      0   0  1
  [32] .debug_line       PROGBITS        00000000 000ef0 0001de 00      0   0  1
  [33] .debug_str        PROGBITS        00000000 0010ce 0000f0 01  MS  0   0  1
  [34] .shstrtab         STRTAB          00000000 0011be 00013b 00      0   0  1
  [35] .symtab           SYMTAB          00000000 0018c4 000c90 10     36  65  4
  [36] .strtab           STRTAB          00000000 002554 000909 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings)
  I (info), L (link order), G (group), x (unknown)
  O (extra OS processing required) o (OS specific), p (processor specific)

There are no section groups in this file.
 ...

Symbol table '.symtab' contains 201 entries:
   Num:    Value  Size Type    Bind   Vis      Ndx Name
...
    99: 100109cc 0x2800000 OBJECT  GLOBAL DEFAULT   26 big_big_array
...
   110: 10010960     4 OBJECT  GLOBAL DEFAULT   23 a_string
...
   130: 10010964     4 OBJECT  GLOBAL DEFAULT   23 a_var_with_value
...
   144: 10000430    96 FUNC    GLOBAL DEFAULT   11 main
```

首先，讓我們看看變量big_big_array，顧名思義，它是一個相當大的全局數組。如果我們跳到symbol表，我們可以看到變量位於位置0x100109cc，我們可以將它與列表中的.bss段相關聯，因為它從它下面的0x100109c8開始。注意它的大小，以及它有多大。我們提到過BSS是二進制圖像的標准段，因為要求磁盤上的二進制文件分配10兆字節的空間是愚蠢的，而所有的空間都是0。注意，本節有一種NOBITS類型，這意味著它在磁盤上沒有任何字節。


因此，.bss段是為全局變量定義的，當程序啟動時，這些變量的值應該為零。在我們討論段時，我們已經看到內存大小如何與磁盤上的大小不同;位於.bss段的變量表示在程序啟動時它們的值為零。


a_string變量位於.sdata段，它表示小數據。小數據(以及相應的.sbss段)是一些體系結構上可用的部分，數據可以通過某個已知指針的偏移量到達。這意味著可以將固定值添加到基地址，這樣就可以更快地獲取段中的數據，因為不需要額外的查找和將地址加載到內存中。大多數架構僅限於直接值的大小可以添加到一個寄存器(例如,如果執行指令r1 =添加r2,70,,70是一個直接的價值,而不是說,添加兩個值存儲在寄存器r1 =添加r2,r3),因此只能抵消一定的“小”距離一個地址。我們還可以看到a_var_with_value位於同一個位置。


然而main位於.text段，正如我們所期望的那樣(記住“text”和“code”這兩個名字可以互換使用來指代內存中的程序。


### Section 和 Segment 在一起


> 8.9的例子。Section 和 Segment 

```
$ readelf --segments /bin/ls

Elf file type is EXEC (Executable file)
Entry point 0x100026c0
There are 8 program headers, starting at offset 52

Program Headers:
  Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
  PHDR           0x000034 0x10000034 0x10000034 0x00100 0x00100 R E 0x4
  INTERP         0x000154 0x10000154 0x10000154 0x0000d 0x0000d R   0x1
      [Requesting program interpreter: /lib/ld.so.1]
  LOAD           0x000000 0x10000000 0x10000000 0x14d5c 0x14d5c R E 0x10000
  LOAD           0x014d60 0x10024d60 0x10024d60 0x002b0 0x00b7c RWE 0x10000
  DYNAMIC        0x014f00 0x10024f00 0x10024f00 0x000d8 0x000d8 RW  0x4
  NOTE           0x000164 0x10000164 0x10000164 0x00020 0x00020 R   0x4
  GNU_EH_FRAME   0x014d30 0x10014d30 0x10014d30 0x0002c 0x0002c R   0x4
  GNU_STACK      0x000000 0x00000000 0x00000000 0x00000 0x00000 RWE 0x4

 Section to Segment mapping:
  Segment Sections...
   00
   01     .interp
   02     .interp .note.ABI-tag .hash .dynsym .dynstr .gnu.version .gnu.version_ r .rela.dyn .rela.plt .init .text .fini .rodata .eh_frame_hdr
   03     .data .eh_frame .got2 .dynamic .ctors .dtors .jcr .got .sdata .sbss .p lt .bss
   04     .dynamic
   05     .note.ABI-tag
   06     .eh_frame_hdr
   07
```


例8.9，“節和段”展示了readelf是如何在二進制/bin/ls文件中顯示段和段映射的


跳到輸出的底部，我們可以看到哪些段被移到了哪些段。例如，. INTERP段被放置到INTERP標記的段中。請注意，readelf告訴我們它正在請求解釋器/lib/ld.so.1;這是一個動態鏈接器，運行它是為了准備二進制文件以便執行。


查看這兩個加載段，我們可以看到文本和數據之間的區別。注意，第一個只有“讀”和“執行”權限，而下一個只有“讀”、“寫”和“執行”權限。這些描述了代碼(r/w)和數據(r/w/e)段。


但是數據不應該是可執行的!實際上，在大多數體系結構(例如，最常見的x86)上，數據段不會被標記為具有可執行數據段。但是，上面的示例輸出來自PowerPC機器，它的編程模型略有不同(ABI，見下文)，要求數據段是可執行的[24]。這就是系統程序員的生活，規則就是要被打破的!


另一件有趣的事情是文件大小與代碼段的內存大小相同，但是內存大小大於數據段的文件大小。這來自BSS段，它包含0的全局變量。

> [24] For those that are curious, the PowerPC ABI calls stubs for functions in dynamic libraries directly in the GOT, rather than having them bounce through a separate PLT entry. Thus the processor needs execute permissions for the GOT section, which you can see is embedded in the data segment. This should make sense after reading the dynamic linking chapter!

## ELF可執行文件

當然，可執行文件是ELF格式的主要用途之一。二進制文件中包含操作系統按照預期執行代碼所需的所有內容。


由於可執行文件被設計成在具有唯一地址空間的進程中運行(參見第6章，虛擬內存)，因此代碼可以假設程序的各個段將被加載到內存中。示例8.10“可執行文件的段”顯示了使用readelf工具檢查可執行文件段的示例。我們可以看到需要放置加載段的虛擬地址。我們可以進一步看到,有一段是代碼——它只讀取和執行權限,一個是數據,不出所料,讀和寫權限,但重要的是沒有執行權限(沒有執行權限,即使缺陷允許攻擊者引入任意數據頁面支持它不會有執行權限,和大多數處理器會因此不允許任何執行的代碼頁)。


> 8.10的例子。可執行文件的段

```
$ readelf --segments /bin/ls

Elf file type is EXEC (Executable file)
Entry point 0x100026c0
There are 8 program headers, starting at offset 52

Program Headers:
  Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
  PHDR           0x000034 0x10000034 0x10000034 0x00100 0x00100 R E 0x4
  INTERP         0x000154 0x10000154 0x10000154 0x0000d 0x0000d R   0x1
      [Requesting program interpreter: /lib/ld.so.1]
  LOAD           0x000000 0x10000000 0x10000000 0x14d5c 0x14d5c R E 0x10000
  LOAD           0x014d60 0x10024d60 0x10024d60 0x002b0 0x00b7c RWE 0x10000
  DYNAMIC        0x014f00 0x10024f00 0x10024f00 0x000d8 0x000d8 RW  0x4
  NOTE           0x000164 0x10000164 0x10000164 0x00020 0x00020 R   0x4
  GNU_EH_FRAME   0x014d30 0x10014d30 0x10014d30 0x0002c 0x0002c R   0x4
  GNU_STACK      0x000000 0x00000000 0x00000000 0x00000 0x00000 RWE 0x4

 Section to Segment mapping:
  Segment Sections...
   00
   01     .interp
   02     .interp .note.ABI-tag .hash .dynsym .dynstr .gnu.version .gnu.version_ r .rela.dyn .rela.plt .init .text .fini .rodata .eh_frame_hdr
   03     .data .eh_frame .got2 .dynamic .ctors .dtors .jcr .got .sdata .sbss .p lt .bss
   04     .dynamic
   05     .note.ABI-tag
   06     .eh_frame_hdr
   07
```


程序段必須裝載在這些地址;鏈接器的最後一步是解決大多數重定位(稱為“符號和重定位”的段)，並用假定的絕對地址對它們進行修補——描述重定位的數據在最後的二進制文件中被丟棄，再也沒有找到這些信息的方法。


實際上，可執行程序通常對共享庫或在整個系統中抽像和共享的公共代碼片段具有外部依賴性——示例8.10“可執行文件的段”中幾乎所有令人困惑的段都與共享庫的使用有關。圖書館在第9章“動態鏈接”一節中被稱為“圖書館”，即動態圖書館。


## 庫

開發人員很快就厭倦了從頭開始編寫所有東西，因此計算機科學的第一個發明之一就是庫。


庫僅僅是您可以從程序中調用的函數的集合。顯然圖書館有許多優點,不僅僅是,你可以節省很多時間通過重用其他人已經做過的工作,通常更相信它有較少的錯誤(因為可能很多人使用的庫,你受益於他們發現和修復bug)。庫與可執行文件完全類似，只是庫函數不是直接運行，而是使用可執行文件中的參數調用。


### 靜態庫

使用庫函數最直接的方法是將庫中的目標文件直接鏈接到最終的可執行文件中，就像您自己編譯的那些文件一樣。這樣鏈接時，庫被稱為靜態庫，因為除非重新編譯程序，否則庫將保持不變。


這是使用庫的最直接的方式，因為最終的結果是一個沒有依賴關系的簡單可執行文件。


### 在靜態庫中

靜態庫只是一組對像文件。目標文件保存在存檔中，這導致了它們通常的後綴擴展名。您可以將歸檔看作類似於zip文件，但不進行壓縮。


下面我們將介紹基本靜態庫的創建，並介紹一些使用庫的常用工具。


> 8.11的例子。創建和使用靜態庫

```
$ cat library.c
/* Library Function */
int function(int input)
{
        return input + 10;
}

$ cat library.h
/* Function Definition */
int function(int);

$ cat program.c
#include <stdio.h>
/* Library header file */
#include "library.h"

int main(void)
{
        int d = function(100);

        printf("%d\n", d);
}

$ gcc -c library.c
$ ar rc libtest.a library.o
$ ranlib ./libtest.a
$ nm --print-armap ./libtest.a

Archive index:
function in library.o

library.o:
00000000 T function

$ gcc -L . program.c -ltest -o program

$ ./program
110
```

首先，我們將庫編譯成一個對像文件，就像我們在前一章看到的那樣。


注意，我們在頭文件中定義了庫API。API由庫中函數的函數定義組成;這是為了讓編譯器知道在構建引用庫的對像文件(例如程序)時函數的類型。c，其中#包含頭文件)。


我們創建庫ar(“archive”的縮寫)命令。按照慣例，靜態庫文件名以lib為前綴，並具有擴展名a。c參數告訴程序創建歸檔文件，a告訴歸檔文件將指定的對像文件添加到庫文件.[25]中


接下來，我們使用ranlib應用程序在庫中使用對像文件內容的符號創建一個頭。這有助於編譯器快速引用符號;在我們只有一個的情況下，這個步驟看起來有點多余;然而，大型庫可能有數千個符號，這意味著索引可以顯著加快查找引用的速度。我們使用nm應用程序檢查這個新的標頭。我們看到函數()函數在偏移為0處的函數符號，正如我們所期望的那樣。


然後用-lname向編譯器指定庫，其中name是庫的文件名，沒有前綴lib，我們還為庫提供了額外的搜索目錄，即當前目錄(-L .)，因為默認情況下，當前目錄不會搜索庫。


最後的結果是一個包含新庫的可執行文件。


### 靜態鏈接的缺點

靜態鏈接非常直接，但是也有一些缺點。


主要有兩個缺點;首先，如果庫代碼被更新(比如為了修復一個bug)，你必須將你的程序重新編譯成一個新的可執行文件;其次，系統中使用該庫的每個程序都在其可執行文件中包含一個副本。這是非常低效的(如果您發現一個bug，並且必須重新編譯，就像第一點一樣)。


例如，C庫(glibc)包含在所有程序中，並提供所有常見函數，比如printf。


### 共享庫

共享庫是解決靜態庫帶來的問題的一種優雅方法。共享庫是在運行時為每個需要它的應用程序動態加載的庫。


應用程序只留下需要某個庫的指針，當進行函數調用時，庫被加載到內存中並執行。如果庫已經加載到另一個應用程序中，那麼可以在這兩個應用程序之間共享代碼，從而節省大量的資源。


這個過程稱為動態鏈接，是現代操作系統中最復雜的段之一。因此，我們將在下一章研究動態鏈接過程。

使用ar創建的>[25]檔案除了創建靜態庫外，還會在Linux系統周圍的一些不同地方彈出。一個廣泛使用的應用程序是deb打包格式，它與Debian、Ubuntu和其他一些Linux系統一起使用就是一個例子。debs使用歸檔文件將所有應用程序文件保存在一個包文件中。RedHat RPM包使用另一種類似的格式cpio。當然，將文件放在一起的規範應用程序是tar文件，它是分發源代碼的通用格式。


## 擴展ELF概念


### 調試


傳統上，事後調試的主要方法被稱為核心轉儲。“磁芯”一詞來源於磁芯存儲器原有的物理特性，即利用小磁環的方向來存儲狀態。


因此，核心轉儲只是程序在特定時間運行時的完整快照。然後，可以使用調試器檢查這個轉儲並重構程序狀態。例8.12“創建核心轉儲並與gdb一起使用的示例”顯示了一個示例程序，該程序寫入隨機內存位置以強制崩潰。此時將停止進程，並記錄當前狀態的轉儲。


> 8.12的例子。創建核心轉儲並與gdb一起使用的示例

```
$ cat coredump.c
int main(void) {
	char *foo = (char*)0x12345;
	*foo = 'a';

	return 0;
}

$ gcc -Wall -g -o coredump coredump.c

$ ./coredump
Segmentation fault (core dumped)

$ file ./core
./core: ELF 32-bit LSB core file Intel 80386, version 1 (SYSV), SVR4-style, from './coredump'

$ gdb ./coredump
...
(gdb) core core
[New LWP 31614]
Core was generated by `./coredump'.
Program terminated with signal 11, Segmentation fault.
#0  0x080483c4 in main () at coredump.c:3
3		*foo = 'a';
(gdb)
```
因此，核心轉儲只是另一個ELF文件，調試器可以理解它的各個段，以表示正在運行的程序的各個段。


### 符號和調試信息

如示例8.12“創建核心轉儲並與gdb一起使用的示例”所示，調試器gdb需要原始可執行文件和核心轉儲來重新構造調試會話的環境。注意，最初的可執行文件是用-g標記構建的，它指示編譯器包含所有調試信息。這些額外的調試信息保存在ELF文件的特殊段中。它詳細描述了當前的寄存器值，代碼中使用了哪些變量，變量的大小，數組的長度等等。它通常采用標准DWARF格式(幾乎是同義詞ELF的雙關)。


包括調試信息可以使可執行文件和庫非常大;雖然這些數據並不需要駐留在內存中以實際運行，但它仍然會占用相當大的磁盤空間。因此，通常的過程是從ELF文件中刪除這些信息。雖然可以同時運送剝離文件和未剝離文件，但大多數當前的二進制分發方法都在單獨的文件中提供調試信息。objcopy工具可以用來提取調試信息(——保持-調試)，然後在原始可執行文件中添加一個鏈接到這個剝離的信息(—add-gnu-debuglink)。完成此操作後，原始可執行文件中將出現一個名為.gnu_debuglink的特殊段，它包含一個散列，以便在調試會話啟動時，調試器可以確保將正確的調試信息與正確的可執行文件關聯起來。


> 8.13的例子。使用objcopy將調試信息剝離到單獨的文件中的示例

```
$ gcc -g -shared -o libtest.so libtest.c
$ objcopy --only-keep-debug libtest.so libtest.debug
$ objcopy --add-gnu-debuglink=libtest.debug libtest.so
$ objdump -s -j .gnu_debuglink libtest.so

libtest.so:     file format elf32-i386

Contents of section .gnu_debuglink:
 0000 6c696274 6573742e 64656275 67000000  libtest.debug...
 0010 52a7fd0a                             R... 
```


符號占用的空間要小得多，但也是從最終輸出中刪除的目標。一旦可執行文件的單個目標文件鏈接到單個最終圖像中，一般就不需要保留大多數符號。正如在“符號和重定位”一節中所討論的那樣，修復重定位條目需要使用符號，但是一旦完成，這些符號就不再是運行最終程序所必需的了。在Linux上，GNU工具鏈剝離程序提供了刪除符號的選項。請注意，有些符號需要在運行時解析(對於動態鏈接，第9章的重點是動態鏈接)，但是這些符號被放在單獨的動態符號表中，這樣它們就不會被刪除，最終的輸出也就沒有用了。


###  coredumps內部

coredump實際上只是另一個ELF文件;這說明了ELF作為二進制格式的靈活性。


> 8.14的例子。使用readelf和eu-readelf檢查coredump的示例。

```
$ readelf --all ./core
ELF Header:
  Magic:   7f 45 4c 46 01 01 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF32
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              CORE (Core file)
  Machine:                           Intel 80386
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          52 (bytes into file)
  Start of section headers:          0 (bytes into file)
  Flags:                             0x0
  Size of this header:               52 (bytes)
  Size of program headers:           32 (bytes)
  Number of program headers:         15
  Size of section headers:           0 (bytes)
  Number of section headers:         0
  Section header string table index: 0

There are no sections in this file.

There are no sections to group in this file.

Program Headers:
  Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
  NOTE           0x000214 0x00000000 0x00000000 0x0022c 0x00000     0
  LOAD           0x001000 0x08048000 0x00000000 0x01000 0x01000 R E 0x1000
  LOAD           0x002000 0x08049000 0x00000000 0x01000 0x01000 RW  0x1000
  LOAD           0x003000 0x489fc000 0x00000000 0x01000 0x1b000 R E 0x1000
  LOAD           0x004000 0x48a17000 0x00000000 0x01000 0x01000 R   0x1000
  LOAD           0x005000 0x48a18000 0x00000000 0x01000 0x01000 RW  0x1000
  LOAD           0x006000 0x48a1f000 0x00000000 0x01000 0x153000 R E 0x1000
  LOAD           0x007000 0x48b72000 0x00000000 0x00000 0x01000     0x1000
  LOAD           0x007000 0x48b73000 0x00000000 0x02000 0x02000 R   0x1000
  LOAD           0x009000 0x48b75000 0x00000000 0x01000 0x01000 RW  0x1000
  LOAD           0x00a000 0x48b76000 0x00000000 0x03000 0x03000 RW  0x1000
  LOAD           0x00d000 0xb771c000 0x00000000 0x01000 0x01000 RW  0x1000
  LOAD           0x00e000 0xb774d000 0x00000000 0x02000 0x02000 RW  0x1000
  LOAD           0x010000 0xb774f000 0x00000000 0x01000 0x01000 R E 0x1000
  LOAD           0x011000 0xbfeac000 0x00000000 0x22000 0x22000 RW  0x1000

There is no dynamic section in this file.

There are no relocations in this file.

There are no unwind sections in this file.

No version information found in this file.

Notes at offset 0x00000214 with length 0x0000022c:
  Owner                 Data size	Description
  CORE                 0x00000090	NT_PRSTATUS (prstatus structure)
  CORE                 0x0000007c	NT_PRPSINFO (prpsinfo structure)
  CORE                 0x000000a0	NT_AUXV (auxiliary vector)
  LINUX                0x00000030	Unknown note type: (0x00000200)

$ eu-readelf -n ./core

Note segment of 556 bytes at offset 0x214:
  Owner          Data size  Type
  CORE                 144  PRSTATUS
    info.si_signo: 11, info.si_code: 0, info.si_errno: 0, cursig: 11
    sigpend: <>
    sighold: <>
    pid: 31614, ppid: 31544, pgrp: 31614, sid: 31544
    utime: 0.000000, stime: 0.000000, cutime: 0.000000, cstime: 0.000000
    orig_eax: -1, fpvalid: 0
    ebx:     1219973108  ecx:     1243440144  edx:              1
    esi:              0  edi:              0  ebp:     0xbfecb828
    eax:          74565  eip:     0x080483c4  eflags:  0x00010286
    esp:     0xbfecb818
    ds: 0x007b  es: 0x007b  fs: 0x0000  gs: 0x0033  cs: 0x0073  ss: 0x007b
  CORE                 124  PRPSINFO
    state: 0, sname: R, zomb: 0, nice: 0, flag: 0x00400400
    uid: 1000, gid: 1000, pid: 31614, ppid: 31544, pgrp: 31614, sid: 31544
    fname: coredump, psargs: ./coredump 
  CORE                 160  AUXV
    SYSINFO: 0xb774f414
    SYSINFO_EHDR: 0xb774f000
    HWCAP: 0xafe8fbff  <fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov clflush dts acpi mmx fxsr sse sse2 ss tm pbe>
    PAGESZ: 4096
    CLKTCK: 100
    PHDR: 0x8048034
    PHENT: 32
    PHNUM: 8
    BASE: 0
    FLAGS: 0
    ENTRY: 0x8048300
    UID: 1000
    EUID: 1000
    GID: 1000
    EGID: 1000
    SECURE: 0
    RANDOM: 0xbfecba1b
    EXECFN: 0xbfecdff1
    PLATFORM: 0xbfecba2b
    NULL
  LINUX                 48  386_TLS
    index: 6, base: 0xb771c8d0, limit: 0x000fffff, flags: 0x00000051
    index: 7, base: 0x00000000, limit: 0x00000000, flags: 0x00000028
    index: 8, base: 0x00000000, limit: 0x00000000, flags: 0x00000028
```

在示例8.14中，“使用readelf和eu-readelf檢查coredump的示例。我們可以看到示例8.12生成的核心文件，“使用readelf工具創建核心轉儲並與gdb一起使用的示例”。文件中沒有任何段、重定位或其他無關信息可能需要加載可執行文件或庫;它只是由一系列描述加載段的程序頭組成。這些段是由內核創建的當前內存分配的原始數據轉儲。


核心轉儲的另一個組件是notes段，其中包含調試所需的數據，但不一定在內存分配的直接快照中捕獲。圖的第二部分中使用的eu-readelf程序通過對數據進行解碼，提供了一個更完整的數據視圖。


PRSTATUS記錄提供了運行過程的一系列有趣信息;例如，我們可以從cursig中看到，程序收到了一個信號11，或分割錯誤，正如我們所期望的。除了進程號信息，它還包括所有當前寄存器的轉儲。給定寄存器值，調試器可以重構堆棧狀態，從而提供回溯;結合來自原始二進制文件的符號和調試信息，調試器可以精確地顯示您如何到達當前執行點。


另一個有趣的輸出是當前輔助向量(AUXV)，在“內核到程序的通信”一節中討論過。386_TLS描述用於線程本地存儲x86實現的全局描述符表條目(有關分段使用的更多信息，請參閱“快速系統調用”一節，有關線程[26]的信息，請參閱“線程”一節)。


內核創建核心轉儲文件的範圍內當前ulimit設置——因為一個程序使用了大量的內存可能會導致一個非常大的垃圾場,可能填滿磁盤,使問題更糟糕的是,通常ulimit設置低甚至為零,因為大多數非開發人員沒有使用核心轉儲文件。但是，在事後分析的方式中，核心轉儲仍然是調試意外情況的最有效的方法。


### 自定義部分

在大多數情況下，代碼、數據和符號的組織是程序員可以保留的工具鏈默認值。然而，有時候擴展或定制部分及其內容是有意義的。一個常見的例子是Linux內核模塊，它用於將驅動程序和其他特性動態加載到正在運行的內核中。因為這些模塊是不可移植的，因為它們只能在一個固定的內核構建版本中工作，所以模塊和內核之間的接口可以是靈活的，不受特定標准的約束。這意味著為moudule存儲許可信息、作者身份、依賴關系和參數等內容的方法可以由內核惟一和完全地定義。


modinfo工具可以在模塊中檢查此信息並將其呈現給用戶。下面我們使用FUSE Linux內核模塊的示例，該模塊允許用戶空間庫向內核提供文件系統實現。


> 8.15的例子。modinfo輸出示例

```
  1 $ cd /lib/modules/$(uname -r)
    
    $ sudo modinfo ./kernel/fs/fuse/fuse.ko 
    filename:       /lib/modules/3.2.0-4-amd64/./kernel/fs/fuse/fuse.ko
  5 alias:          devname:fuse
    alias:          char-major-10-229
    license:        GPL
    description:    Filesystem in Userspace
    author:         Miklos Szeredi <miklos@szeredi.hu>
 10 depends:        
    intree:         Y
    vermagic:       3.2.0-4-amd64 SMP mod_unload modversions 
    parm:           max_user_bgreq:Global limit for the maximum number of backgrounded requests an unprivileged user can set (uint)
    parm:           max_user_congthresh:Global limit for the maximum congestion threshold an unprivileged user can set (uint)
 15 
    $ objdump -s -j .modinfo ./kernel/fs/fuse/fuse.ko 
    
    ./kernel/fs/fuse/fuse.ko:     file format elf64-x86-64
    
 20 Contents of section .modinfo:
     0000 616c6961 733d6465 766e616d 653a6675  alias=devname:fu
     0010 73650061 6c696173 3d636861 722d6d61  se.alias=char-ma
     0020 6a6f722d 31302d32 32390070 61726d3d  jor-10-229.parm=
     0030 6d61785f 75736572 5f636f6e 67746872  max_user_congthr
 25  0040 6573683a 476c6f62 616c206c 696d6974  esh:Global limit
     0050 20666f72 20746865 206d6178 696d756d   for the maximum
     0060 20636f6e 67657374 696f6e20 74687265   congestion thre
     0070 73686f6c 6420616e 20756e70 72697669  shold an unprivi
     0080 6c656765 64207573 65722063 616e2073  leged user can s
 30  0090 65740070 61726d74 7970653d 6d61785f  et.parmtype=max_
     00a0 75736572 5f636f6e 67746872 6573683a  user_congthresh:
     00b0 75696e74 00706172 6d3d6d61 785f7573  uint.parm=max_us
     00c0 65725f62 67726571 3a476c6f 62616c20  er_bgreq:Global 
     00d0 6c696d69 7420666f 72207468 65206d61  limit for the ma
 35  00e0 78696d75 6d206e75 6d626572 206f6620  ximum number of 
     00f0 6261636b 67726f75 6e646564 20726571  backgrounded req
     0100 75657374 7320616e 20756e70 72697669  uests an unprivi
     0110 6c656765 64207573 65722063 616e2073  leged user can s
     0120 65740070 61726d74 7970653d 6d61785f  et.parmtype=max_
 40  0130 75736572 5f626772 65713a75 696e7400  user_bgreq:uint.
     0140 6c696365 6e73653d 47504c00 64657363  license=GPL.desc
     0150 72697074 696f6e3d 46696c65 73797374  ription=Filesyst
     0160 656d2069 6e205573 65727370 61636500  em in Userspace.
     0170 61757468 6f723d4d 696b6c6f 7320537a  author=Miklos Sz
 45  0180 65726564 69203c6d 696b6c6f 7340737a  eredi <miklos@sz
     0190 65726564 692e6875 3e000000 00000000  eredi.hu>.......
     01a0 64657065 6e64733d 00696e74 7265653d  depends=.intree=
     01b0 59007665 726d6167 69633d33 2e322e30  Y.vermagic=3.2.0
     01c0 2d342d61 6d643634 20534d50 206d6f64  -4-amd64 SMP mod
 50  01d0 5f756e6c 6f616420 6d6f6476 65727369  _unload modversi
     01e0 6f6e7320 00                          ons .           
```

如上所示，modinfo正在解析模塊文件中嵌入的.modinfo段，以顯示模塊的詳細信息。例8.16“將模塊信息放入段”展示了如何將“author”字段放入模塊中。代碼主要來自include/linux/module.h。


> 8.16的例子。將模塊信息分成幾個部分

```
  1 /*
     * Start at the bottom, and work your way up!
     */
    
  5 /* Indirect macros required for expanded argument pasting, eg. __LINE__. */
    #define ___PASTE(a,b) a##b
    #define __PASTE(a,b) ___PASTE(a,b)
    
    
 10 #define __UNIQUE_ID(prefix) __PASTE(__PASTE(__UNIQUE_ID_, prefix), __COUNTER__)
    
    /* Indirect stringification.  Doing two levels allows the parameter to be a
     * macro itself.  For example, compile with -DFOO=bar, __stringify(FOO)
     * converts to "bar".
 15  */
    
    #define __stringify_1(x...)     #x
    #define __stringify(x...)       __stringify_1(x)
    
 20 #define __MODULE_INFO(tag, name, info)                                    \
    static const char __UNIQUE_ID(name)[]                                     \
      __used __attribute__((section(".modinfo"), unused, aligned(1)))         \
      = __stringify(tag) "=" info
    
 25 /* Generic info of form tag = "info" */
    #define MODULE_INFO(tag, info) __MODULE_INFO(tag, tag, info)
    
    /*
     * Author(s), use "Name <email>" or just "Name", for multiple
 30  * authors use multiple MODULE_AUTHOR() statements/lines.
     */
    #define MODULE_AUTHOR(_author) MODULE_INFO(author, _author)
    
    /* ---- */
 35 
    MODULE_AUTHOR("Your Name <your@name.com>");
```

起初，這看起來像是一場宏觀噩夢，但它可以逐步被解開。從底部開始，我們看到MODULE_AUTHOR是更通用的“__MODULE_INFO”宏的包裝器，這是大多數魔術發生的地方。在這裡，我們可以看到我們正在構建一個靜態const char[]變量來保存字符串“author=Your Name ' '”。值得注意的是，這個變量有一個額外的參數`__attribute__((section(".modinfo")))`，它告訴編譯器不要將它與所有其他變量一起放入數據部分，而是將其存儲在名為.modinfo的ELF部分中。其他參數阻止了變量的優化，因為它看起來是未使用的，並且通過指定對齊方式確保我們將變量打包在一起。


大量使用了stringification宏，這是在C預處理器中使用的相當神秘的技巧，以確保字符串和定義可以共存。唯一的另一個技巧是使用gcc提供的`__COUNTER__` 特殊定義，它在每次調用時都提供一個惟一的遞增值;這允許在一個文件中調用多個MODULE_AUTHOR，而不會以相同的變量名結束。


我們可以檢查放置在最終模塊中的符號，查看最終結果:


> 8.17的例子。模塊符號。modinfo部分
```
  1 $ objdump --syms ./fuse.ko | grep modinfo
    
    0000000000000000 l    d  .modinfo	0000000000000000 .modinfo
    0000000000000000 l     O .modinfo	0000000000000013 __UNIQUE_ID_alias1
  5 0000000000000013 l     O .modinfo	0000000000000018 __UNIQUE_ID_alias0
    000000000000002b l     O .modinfo	0000000000000011 __UNIQUE_ID_alias8
    000000000000003c l     O .modinfo	000000000000000e __UNIQUE_ID_alias7
    000000000000004a l     O .modinfo	0000000000000068 __UNIQUE_ID_max_user_congthresh6
    00000000000000b2 l     O .modinfo	0000000000000022 __UNIQUE_ID_max_user_congthreshtype5
 10 00000000000000d4 l     O .modinfo	000000000000006e __UNIQUE_ID_max_user_bgreq4
    0000000000000142 l     O .modinfo	000000000000001d __UNIQUE_ID_max_user_bgreqtype3
    000000000000015f l     O .modinfo	000000000000000c __UNIQUE_ID_license2
    000000000000016b l     O .modinfo	0000000000000024 __UNIQUE_ID_description1
    000000000000018f l     O .modinfo	000000000000002a __UNIQUE_ID_author0
 15 00000000000001b9 l     O .modinfo	0000000000000011 __UNIQUE_ID_alias0
    00000000000001d0 l     O .modinfo	0000000000000009 __module_depends
    00000000000001d9 l     O .modinfo	0000000000000009 __UNIQUE_ID_intree1
    00000000000001e2 l     O .modinfo	000000000000002f __UNIQUE_ID_vermagic0
```

### 鏈接器腳本

在示例8.7中，我們描述了段如何在最終輸出中組成部分。鏈接器的工作是將這些段構建成片段;為了實現這一點，它使用了一個鏈接器腳本，該腳本描述了段的起始位置、進入段的部分以及各種其他參數。


例8.18“默認鏈接器腳本”顯示了默認鏈接器腳本的一個摘錄，當鏈接器通過指定-Wl、-verbose到gcc來指定其詳細標志時，將顯示該腳本。默認腳本內置在鏈接器中，基於標准API定義為構建平台創建可用的用戶空間程序。


> 8.18的例子。默認鏈接器腳本

```
  1 $ gcc -Wl,--verbose -o test test.c
    GNU ld (GNU Binutils for Debian) 2.26
    ...
    using internal linker script:
  5 ==================================================
    OUTPUT_FORMAT("elf64-x86-64", "elf64-x86-64",
    	      "elf64-x86-64")
    OUTPUT_ARCH(i386:x86-64)
    ENTRY(_start)
 10 SEARCH_DIR("=/usr/local/lib/x86_64-linux-gnu"); ...
    SECTIONS
    {
      /* Read-only sections, merged into text segment: */
      PROVIDE (__executable_start = SEGMENT_START("text-segment", 0x400000)); . = SEGMENT_START("text-segment", 0x400000) + SIZEOF_HEADERS;
 15   .interp         : { *(.interp) }
      .note.gnu.build-id : { *(.note.gnu.build-id) }
      .hash           : { *(.hash) }
      .gnu.hash       : { *(.gnu.hash) }
      .dynsym         : { *(.dynsym) }
 20   .dynstr         : { *(.dynstr) }
      .gnu.version    : { *(.gnu.version) }
      .gnu.version_d  : { *(.gnu.version_d) }
      .gnu.version_r  : { *(.gnu.version_r) }
      .rela.dyn       :
 25     {
        ...
        }
      PROVIDE (etext = .);
      .rodata         : { *(.rodata .rodata.* .gnu.linkonce.r.*) }
 30   .rodata1        : { *(.rodata1) }
    ...
```

您可以大致看到鏈接器腳本如何指定開始位置以及將哪些段分組到各個部分。與使用-Wl通過gcc將-verbose傳遞給鏈接器的方式相同，標記可以提供定制的鏈接器腳本。普通的用戶空間開發人員不太可能需要覆蓋默認的鏈接器腳本。然而，通常非常定制的應用程序(如內核構建)需要定制的鏈接器腳本。


對於多線程應用程序，每個運行的線程都會有重復的條目。調試器會理解這一點，gdb就是這樣實現線程命令來顯示和在線程之間切換的。


## ABI

ABI是您在使用系統編程時經常聽到的術語。我們已經詳細討論了API，它是程序員對代碼的接口。


ABI是指較低級別的接口，編譯器、操作系統以及某種程度上的處理器必須同意這些接口進行通信。下面我們將介紹一些概念，這些概念對於理解ABI注意事項非常重要。


### 字節順序

Endianess


### 調用約定


#### 傳遞參數


寄存器或堆棧?


#### 功能描述符

在許多體系結構中，必須通過函數描述符調用函數，而不是直接調用函數。


例如，在IA64上，函數描述符包含兩個組件;函數的地址(64位或8字節值)和全局指針(gp)的地址。ABI指定r1應該總是包含一個函數的gp值。這意味著，當您調用一個函數時，調用函數的任務是保存它們的gp值，將r1設置為新值(來自函數描述符)，然後調用函數。


這看起來似乎是一種奇怪的方法，但它具有非常有用的實際意義，您將在下一章看到全局偏移表。在IA64上，添加指令只能接受最大的22位立即值[27]。立即值是直接指定的值，而不是在寄存器中指定的值(例如，在add r1 + 100100中是立即值)。


您可能認為22位可以表示4194304字節或4MB。因此，每個函數都可以直接偏移到4MB大的內存區域，而無需將任何值加載到寄存器中。如果編譯器、鏈接器和加載器都同意全局指針指向什麼(如ABI中指定的那樣)，那麼通過減少加載就可以提高性能。


從技術上講，這是因為IA64捆綁指令的方式。在每個包中放入三個指令，並且只有足夠的空間來保持一個22位的值來保持包在一起。



#### 開始一個進程

我們之前提到過，簡單地說程序從main()函數開始並不完全正確。下面我們將分析在加載和運行一個典型的動態鏈接程序時發生了什麼

<!-- (靜態鏈接的程序是相似的，但是不同的XXX，我們應該對此進行討論嗎?) -->


首先，為了響應exec系統調用，內核為新進程分配結構並從磁盤讀取指定的ELF文件。


我們提到ELF有一個程序解釋器字段PT_INTERP，它可以被設置為“解釋”程序。對於動態鏈接的應用程序，解釋器是動態鏈接器，也就是ld.so，它允許在程序啟動之前動態地完成一些鏈接過程。


在這種情況下，內核還讀取動態鏈接器代碼，並按照它指定的入口點地址啟動程序。我們檢查的作用深度動態鏈接器在下一章,但足以說等設置加載應用程序所需的任何庫中指定(二進制)的動態部分,然後開始執行程序的二進制的入口點地址(即_init函數)。


#### 與程序的內核通信

內核需要在程序啟動時與程序進行通信;即程序的參數、當前環境變量和稱為輔助向量或auxv的特殊結構(您可以通過指定環境值LD_SHOW_AUXV=1請求動態鏈接器顯示auxv的一些調試輸出)。


參數和環境非常直接，exec系統調用的各種形式允許您為程序指定這些參數。


內核通過將所有需要的信息放在堆棧上進行通信，以便新創建的程序接收這些信息。因此，當程序啟動時，它可以使用它的堆棧指針來查找所需的所有啟動信息。


輔助向量是一種特殊的結構，用於將信息直接從內核傳遞到新運行的程序。它包含可能需要的系統特定信息，例如系統上虛擬內存頁面的默認大小或硬件功能;這是內核識別的特定特性，用戶空間程序可以利用底層硬件。

### 內核庫

我們在前面提到過，系統調用很慢，現代系統具有避免調用處理器陷阱的開銷的機制。


在Linux中，這是通過動態加載器和內核之間的一個巧妙的技巧實現的，這些技巧都與AUXV結構通信。內核實際上是在每個新創建的進程的地址空間中添加了一個小的共享庫，其中包含一個為您進行系統調用的函數。這個系統的美妙之處在於，如果底層硬件支持快速系統調用機制，內核(作為庫的創建者)可以使用它，否則它可以使用生成陷阱的舊方案。這個庫被命名為linux-gate。之所以叫它，是因為它是通向內核內部工作的通道。


當內核啟動動態鏈接器時，它向auxv添加一個名為AT_SYSINFO_EHDR的條目，這是內存中的地址，特殊的內核庫就位於這個地址中。當動態鏈接器啟動時，它可以查找AT_SYSINFO_EHDR指針，如果找到，則為程序加載該庫。程序不知道這個庫存在;這是動態鏈接器和內核之間的私有安排。


我們提到程序員通過調用系統庫中的函數間接地進行系統調用，即libc。libc可以檢查是否加載了特殊的內核二進制文件，如果是，則使用其中的函數進行系統調用。如前所述，如果內核確定硬件的能力，這將使用快速系統調用方法。


### 啟動程序

一旦內核加載了解釋器，它就會把它傳遞到解釋器文件中給出的入口點(注意，這裡不會檢查動態鏈接器是如何啟動的;有關動態鏈接的完整討論，請參閱第9章，動態鏈接)。動態鏈接器將跳轉到ELF二進制文件中給出的入口地址。


> 8.19的例子。反彙編程序啟動

```
  1 $ cat test.c
    
    int main(void)
    {
  5 	return 0;
    }
    
    $ gcc -o test test.c
    
 10 $ readelf --headers ./test | grep Entry
      Entry point address:               0x80482b0
    
    $ objdump --disassemble ./test
    
 15 [...]
    
    080482b0 <_start>:
     80482b0:       31 ed                   xor    %ebp,%ebp
     80482b2:       5e                      pop    %esi
 20  80482b3:       89 e1                   mov    %esp,%ecx
     80482b5:       83 e4 f0                and    $0xfffffff0,%esp
     80482b8:       50                      push   %eax
     80482b9:       54                      push   %esp
     80482ba:       52                      push   %edx
 25  80482bb:       68 00 84 04 08          push   $0x8048400
     80482c0:       68 90 83 04 08          push   $0x8048390
     80482c5:       51                      push   %ecx
     80482c6:       56                      push   %esi
     80482c7:       68 68 83 04 08          push   $0x8048368
 30  80482cc:       e8 b3 ff ff ff          call   8048284 <__libc_start_main@plt>
     80482d1:       f4                      hlt
     80482d2:       90                      nop
     80482d3:       90                      nop
    
 35 08048368 <main>:
     8048368:       55                      push   %ebp
     8048369:       89 e5                   mov    %esp,%ebp
     804836b:       83 ec 08                sub    $0x8,%esp
     804836e:       83 e4 f0                and    $0xfffffff0,%esp
 40  8048371:       b8 00 00 00 00          mov    $0x0,%eax
     8048376:       83 c0 0f                add    $0xf,%eax
     8048379:       83 c0 0f                add    $0xf,%eax
     804837c:       c1 e8 04                shr    $0x4,%eax
     804837f:       c1 e0 04                shl    $0x4,%eax
 45  8048382:       29 c4                   sub    %eax,%esp
     8048384:       b8 00 00 00 00          mov    $0x0,%eax
     8048389:       c9                      leave
     804838a:       c3                      ret
     804838b:       90                      nop
 50  804838c:       90                      nop
     804838d:       90                      nop
     804838e:       90                      nop
     804838f:       90                      nop
    
 55 08048390 <__libc_csu_init>:
     8048390:       55                      push   %ebp
     8048391:       89 e5                   mov    %esp,%ebp
     [...]
    
 60 08048400 <__libc_csu_fini>:
     8048400:       55                      push   %ebp
     [...]
```

上面我們研究了最簡單的程序。使用readelf，我們可以看到入口點是二進制文件中的_start函數。此時，我們可以在反彙編層看到一些值被推到堆棧上。第一個值0x8048400是 `__libc_csu_fini` 函數;0x8048390是`__libc_csu_init`，最後是0x8048368, main()函數。然後調用`__libc_start_main`函數。


`__libc_start_main`是在glibc源sysdeps/generic/libc-start.c中定義的。文件函數非常復雜，隱藏在大量定義之間，因為它需要在glibc可以運行的非常廣泛的系統和體系結構之間進行移植。它做了許多與建立C庫相關的具體事情，而這是一般程序員不需要擔心的。庫回調到程序中的下一個點是處理init代碼。


init和fini是兩個特殊的概念，它們分別調用共享庫中可能需要在庫啟動之前或庫被卸載時調用的代碼部分。您可以看到，對於庫程序員來說，在庫啟動時設置變量或者在庫結束時清理變量是非常有用的。最初在庫中查找函數_init和_fini;然而，由於在這些函數中所有的東西都是必需的，這就有點限制了。下面我們將研究init/fini過程是如何工作的。


在這個階段，我們可以看到`__libc_start_main`函數將在堆棧上接收到相當多的輸入參數。首先，它可以訪問內核中的程序參數、環境變量和輔助向量。然後，初始化函數將被推送到堆棧地址，以便函數處理init、fini，最後是主函數本身的地址。


我們需要在源代碼中指明函數應該由init或fini調用。在gcc中，我們使用屬性在主程序中將兩個函數標記為構造函數和析構函數。這些術語通常與面向對像語言一起用於描述對像生命周期。


> 8.20的例子。構造函數和析構函數

```
  1 $ cat test.c
    #include <stdio.h>
    
    void __attribute__((constructor)) program_init(void)  {
  5   printf("init\n");
    }
    
    void  __attribute__((destructor)) program_fini(void) {
      printf("fini\n");
 10 }
    
    int main(void)
    {
      return 0;
 15 }
    
    $ gcc -Wall  -o test test.c
    
    $ ./test
 20 init
    fini
    
    $ objdump --disassemble ./test | grep program_init
    08048398 <program_init>:
 25 
    $ objdump --disassemble ./test | grep program_fini
    080483b0 <program_fini>:
    
    $ objdump --disassemble ./test 
 30 
    [...]
    08048280 <_init>:
     8048280:       55                      push   %ebp
     8048281:       89 e5                   mov    %esp,%ebp
 35  8048283:       83 ec 08                sub    $0x8,%esp
     8048286:       e8 79 00 00 00          call   8048304 <call_gmon_start>
     804828b:       e8 e0 00 00 00          call   8048370 <frame_dummy>
     8048290:       e8 2b 02 00 00          call   80484c0 <__do_global_ctors_aux>
     8048295:       c9                      leave
 40  8048296:       c3                      ret
    [...]
    
    080484c0 <__do_global_ctors_aux>:
     80484c0:       55                      push   %ebp
 45  80484c1:       89 e5                   mov    %esp,%ebp
     80484c3:       53                      push   %ebx
     80484c4:       52                      push   %edx
     80484c5:       a1 2c 95 04 08          mov    0x804952c,%eax
     80484ca:       83 f8 ff                cmp    $0xffffffff,%eax
 50  80484cd:       74 1e                   je     80484ed <__do_global_ctors_aux+0x2d>
     80484cf:       bb 2c 95 04 08          mov    $0x804952c,%ebx
     80484d4:       8d b6 00 00 00 00       lea    0x0(%esi),%esi
     80484da:       8d bf 00 00 00 00       lea    0x0(%edi),%edi
     80484e0:       ff d0                   call   *%eax
 55  80484e2:       8b 43 fc                mov    0xfffffffc(%ebx),%eax
     80484e5:       83 eb 04                sub    $0x4,%ebx
     80484e8:       83 f8 ff                cmp    $0xffffffff,%eax
     80484eb:       75 f3                   jne    80484e0 <__do_global_ctors_aux+0x20>
     80484ed:       58                      pop    %eax
 60  80484ee:       5b                      pop    %ebx
     80484ef:       5d                      pop    %ebp
     80484f0:       c3                      ret
     80484f1:       90                      nop
     80484f2:       90                      nop
 65  80484f3:       90                      nop
    
    
    $ readelf --sections ./test
    There are 34 section headers, starting at offset 0xfb0:
 70 
    Section Headers:
      [Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
      [ 0]                   NULL            00000000 000000 000000 00      0   0  0
      [ 1] .interp           PROGBITS        08048114 000114 000013 00   A  0   0  1
 75   [ 2] .note.ABI-tag     NOTE            08048128 000128 000020 00   A  0   0  4
      [ 3] .hash             HASH            08048148 000148 00002c 04   A  4   0  4
      [ 4] .dynsym           DYNSYM          08048174 000174 000060 10   A  5   1  4
      [ 5] .dynstr           STRTAB          080481d4 0001d4 00005e 00   A  0   0  1
      [ 6] .gnu.version      VERSYM          08048232 000232 00000c 02   A  4   0  2
 80   [ 7] .gnu.version_r    VERNEED         08048240 000240 000020 00   A  5   1  4
      [ 8] .rel.dyn          REL             08048260 000260 000008 08   A  4   0  4
      [ 9] .rel.plt          REL             08048268 000268 000018 08   A  4  11  4
      [10] .init             PROGBITS        08048280 000280 000017 00  AX  0   0  4
      [11] .plt              PROGBITS        08048298 000298 000040 04  AX  0   0  4
 85   [12] .text             PROGBITS        080482e0 0002e0 000214 00  AX  0   0 16
      [13] .fini             PROGBITS        080484f4 0004f4 00001a 00  AX  0   0  4
      [14] .rodata           PROGBITS        08048510 000510 000012 00   A  0   0  4
      [15] .eh_frame         PROGBITS        08048524 000524 000004 00   A  0   0  4
      [16] .ctors            PROGBITS        08049528 000528 00000c 00  WA  0   0  4
 90   [17] .dtors            PROGBITS        08049534 000534 00000c 00  WA  0   0  4
      [18] .jcr              PROGBITS        08049540 000540 000004 00  WA  0   0  4
      [19] .dynamic          DYNAMIC         08049544 000544 0000c8 08  WA  5   0  4
      [20] .got              PROGBITS        0804960c 00060c 000004 04  WA  0   0  4
      [21] .got.plt          PROGBITS        08049610 000610 000018 04  WA  0   0  4
 95   [22] .data             PROGBITS        08049628 000628 00000c 00  WA  0   0  4
      [23] .bss              NOBITS          08049634 000634 000004 00  WA  0   0  4
      [24] .comment          PROGBITS        00000000 000634 00018f 00      0   0  1
      [25] .debug_aranges    PROGBITS        00000000 0007c8 000078 00      0   0  8
      [26] .debug_pubnames   PROGBITS        00000000 000840 000025 00      0   0  1
100   [27] .debug_info       PROGBITS        00000000 000865 0002e1 00      0   0  1
      [28] .debug_abbrev     PROGBITS        00000000 000b46 000076 00      0   0  1
      [29] .debug_line       PROGBITS        00000000 000bbc 0001da 00      0   0  1
      [30] .debug_str        PROGBITS        00000000 000d96 0000f3 01  MS  0   0  1
      [31] .shstrtab         STRTAB          00000000 000e89 000127 00      0   0  1
105   [32] .symtab           SYMTAB          00000000 001500 000490 10     33  53  4
      [33] .strtab           STRTAB          00000000 001990 000218 00      0   0  1
    Key to Flags:
      W (write), A (alloc), X (execute), M (merge), S (strings)
      I (info), L (link order), G (group), x (unknown)
110   O (extra OS processing required) o (OS specific), p (processor specific)
    
    $ objdump --disassemble-all --section .ctors ./test
    
    ./test:     file format elf32-i386
115 
    Contents of section .ctors:
     8049528 ffffffff 98830408 00000000           ............
```

__libc_start_main的最後一個推入堆棧的值是初始化函數__libc_csu_init。如果我們沿著__libc_csu_init的調用鏈，我們可以看到它做了一些設置，然後在可執行文件中調用_init函數。_init函數最終調用一個名為__do_global_ctors_aux的函數。查看這個函數的反彙編層，我們可以看到它似乎從地址0x804952c開始循環，讀取一個值並調用它。我們可以看到這個起始地址在文件的.ctors段;如果我們深入查看，我們會發現它包含第一個值-1，一個函數地址(大端字節格式)和值0。


大endian格式的地址是0x08048398，也就是program_init函數的地址!因此，.ctors段的格式首先是-1，然後是初始化時要調用的函數地址，最後是一個0表示列表已經完成。每個條目都將被調用(在本例中，我們只有一個函數)。


一旦__libc_start_main完成了_init調用，它最終會調用main()函數!請記住，它最初是使用內核的參數和環境指針進行堆棧設置的;這就是main獲取argc, argv[]， envp[]參數的方式。現在運行這個過程，設置階段就完成了。


當程序退出時，用.dtors實現了一個類似的過程。__libc_start_main在main()函數完成時調用這些函數。


正如您所看到的，在程序開始之前已經做了很多事情，甚至在您認為它已經完成之後也做了一點!