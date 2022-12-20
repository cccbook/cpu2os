# 第7章。工具鏈


## 編譯v解釋程序


### 編譯程序

到目前為止，我們已經討論了如何將程序加載到虛擬內存中，作為操作系統跟蹤並通過系統調用與之交互的進程啟動。


可以直接加載到內存中的程序需要采用直接的二進制格式。將用C等語言編寫的源代碼轉換為二進制文件以便執行的過程稱為編譯。毫不奇怪，這個過程是由編譯器完成的;最普遍的例子是gcc。


### 解釋程序

編譯程序對於現代軟件開發有一些缺點。每當開發人員進行更改時，必須調用編譯器重新創建可執行文件。它是一個邏輯擴展，可以設計一個編譯程序，它可以讀取另一個程序列表並逐行執行代碼。


我們稱這種類型的編譯程序為解釋器，因為它解釋輸入文件的每一行，並將其作為代碼執行。這樣，程序就不需要編譯，並且在解釋器下一次運行代碼時將看到任何更改。


為了方便起見，解釋程序的運行速度通常比編譯程序慢。程序每次讀取和解釋代碼時的開銷僅在編譯程序中遇到一次，而解釋程序每次運行時都遇到一次。


但是解釋語言有很多積極的方面。許多解釋語言實際上是在從底層硬件抽像出來的虛擬機上運行的。Python和Perl 6是實現解釋代碼運行的虛擬機的語言。


### 虛擬機

編譯後的程序完全依賴於它所編譯的機器的硬件，因為它必須能夠簡單地復制到內存中並執行。虛擬機是將硬件抽像為軟件。


例如，Java是一種混合語言，部分編譯，部分解釋。Java代碼被編譯成運行在Java虛擬機或更通常稱為JVM中的程序。這意味著編譯後的程序可以在任何為其編寫JVM的硬件上運行;所謂寫一個，到處跑。

## 構建可執行文件

當我們討論編譯器時，實際上涉及到創建可執行文件的三個獨立步驟。


1。編譯

2。組譯

3.鏈接


這個過程中涉及的組件統稱為工具鏈，因為工具將其中一個的輸出鏈到另一個的輸入，以創建最終的輸出。


鏈中的每個鏈接使源代碼逐漸接近適合執行的二進制代碼。


## 編譯


編譯過程


將源文件編譯為可執行文件的第一步是將代碼從高級的、可理解的語言轉換為彙編代碼。從前幾章我們知道彙編代碼直接與處理器提供的指令和寄存器工作。


由於許多原因，編譯器是過程中最復雜的步驟。首先，人類是非常不可預測的，他們的源代碼有很多不同的形式。編譯器只對實際代碼感興趣，但是人們需要注釋和空格(空格、制表符、縮進等)來理解代碼。編譯器將人工編寫的源代碼轉換為其內部表示的過程稱為解析。


###  C代碼

對於C代碼，在解析被稱為預處理程序的源代碼之前實際上有一個步驟。預處理程序的核心是文本替換程序。例如，任何聲明為#define變量文本的變量都將被替換為文本。然後將這些經過預處理的代碼傳遞給編譯器。


### 語法

任何計算語言都有描述語言規則的特定語法。您和編譯器都知道語法規則，一切順利的話，您就會相互理解。人類，就像他們一樣，經常忘記或打破規則，導致編譯器無法理解你的意圖。例如，如果要在if條件外保留右括號，編譯器就不知道實際的條件在哪裡。


語法通常用Backus-Naur形式(BNF)[21]來描述，[21]是一種可以用來描述語言的語言!

### 組譯器產生代碼

編譯器的工作是將高級語言轉換成對應機器的彙編代碼。顯然，每個不同的機器都有不同的指令集、不同的寄存器數量和不同的正確操作規則。


### 對齊


> 圖7.1。對齊

![](http://www.bottomupcs.com/chapter06/figures/alignment.png)

> CPU通常只能在特定的對齊方式下將值從內存加載到寄存器中。未對齊的負載最多會導致性能下降。



內存中變量的對齊是編譯器需要考慮的一個重要問題。系統程序員需要知道對齊約束，以幫助編譯器創建它能創建的最高效的代碼。


cpu通常不能從任意內存位置將值加載到寄存器中。它要求變量在一定的邊界上對齊。在上面的示例中，我們可以看到如何將32位(4字節)的值加載到需要對變量進行4字節對齊的機器上的寄存器中。


第一個變量可以直接加載到寄存器中，因為它位於4字節邊界之間。然而，第二個變量跨越了4字節邊界。這意味著至少需要兩次負載才能將變量放入一個寄存器中;首先是下半部分，然後是上半部分。


有些體系結構，比如x86，可以在硬件中處理未對齊的負載，唯一的症狀是性能降低，因為硬件要做額外的工作才能將值放入寄存器。其他體系結構不能違反對齊規則，並且會引發一個異常，操作系統通常會捕獲該異常，然後不得不手動加載部分寄存器，從而導致更多的開銷。


### 結構填充


程序員需要考慮對齊，尤其是在創建結構時。雖然編譯器知道它所構建的體系結構的對齊規則，但有時程序員會導致次優行為。


C99標准只是說，結構在內存中按照聲明中指定的順序排列，而在結構數組中，所有元素的大小都相同。


> 7.1的例子。結構體填充例子

```c
$ cat struct.c
#include <stdio.h>

struct a_struct {
        char char_one;
        char char_two;
        int int_one;
};

int main(void)
{

        struct a_struct s;

        printf("%p : s.char_one\n" \
               "%p : s.char_two\n" \
               "%p : s.int_one\n", &s.char_one,
               &s.char_two, &s.int_one);

        return 0;

}

$ gcc -o struct struct.c

$ gcc -fpack-struct -o struct-packed struct.c

$ ./struct
0x7fdf6798 : s.char_one
0x7fdf6799 : s.char_two
0x7fdf679c : s.int_one

$ ./struct-packed
0x7fcd2778 : s.char_one
0x7fcd2779 : s.char_two
0x7fcd277a : s.int_one
```


在上面的例子中，我們設計了一個有兩個字節(字符後面跟著一個4字節整數)的結構。編譯器對結構進行如下填充。


> 圖7.2。對齊

![](http://www.bottomupcs.com/chapter06/figures/padding.png)

編譯器對結構進行填充，使其在4字節邊界上對齊整數。



在另一個示例中，我們指示編譯器不要填充結構，相應地，我們可以看到整數在兩個字符之後直接開始。


### 高速緩存線對齊

我們之前討論過緩存中的混疊，以及幾個地址如何映射到同一個緩存線。程序員需要確保在編寫程序時不會導致緩存線路的反彈。


當程序不斷訪問映射到同一高速緩存線路的兩個內存區域時，就會出現這種情況。這有效地浪費了高速緩存線，因為它被加載，在短時間內使用，然後必須被踢出，而另一條高速緩存線加載到緩存中的相同位置。


顯然，如果這種情況重復發生，性能將顯著降低。如果以稍微不同的方式組織衝突的數據以避免緩存線衝突，情況將會得到緩解。


檢測這種情況的一種可能的方法是剖析。當您分析您的代碼時，您要“觀察”它，以分析需要哪些代碼路徑以及它們需要多長時間才能執行。使用配置文件引導優化(PGO)，編譯器可以在它構建的第一個二進制文件中添加特殊的額外代碼，該二進制文件運行並記錄所獲取的分支，等等。然後，您可以使用額外的信息重新編譯二進制文件，以創建性能更好的二進制文件。否則，程序員可以查看概要文件的輸出，並可能檢測到諸如緩存線反彈之類的情況。 <!-- (XXX其他地方?) -->


### 空間-速度交換

編譯器在上面所做的工作是通過使用一些額外的內存來獲得運行代碼的速度改進。編譯器知道體系結構的規則，並且能夠決定最佳的數據對齊方式，可能通過犧牲少量的內存浪費來提高(或者甚至只是糾正)性能。


因此，作為一名程序員，您永遠不應該對編譯器布局變量和數據的方式做出假設。這樣做是不可移植的，因為不同的體系結構可能有不同的規則，編譯器可能根據顯式命令或優化級別做出不同的決策。


### 做出假設

因此，作為一名C程序員，您需要熟悉關於編譯器將要做什麼以及什麼是變量的假設。C99標准中詳細列出了您可以假設和不能假設的具體內容;如果您正在用C編程，那麼為了避免編寫不可移植的或有bug的代碼，熟悉規則當然是值得的。


> 7.2的例子。堆棧對齊的例子

```c
$ cat stack.c
#include <stdio.h>

struct a_struct {
        int a;
        int b;
};

int main(void)
{
        int i;
        struct a_struct s;
        printf("%p\n%p\ndiff %ld\n", &i, &s, (unsigned long)&s - (unsigned long)&i);
        return 0;
}
$ gcc-3.3 -Wall -o stack-3.3 ./stack.c
$ gcc-4.0 -o stack-4.0 stack.c

$ ./stack-3.3
0x60000fffffc2b510
0x60000fffffc2b520
diff 16

$ ./stack-4.0
0x60000fffff89b520
0x60000fffff89b524
diff 4
```

在上面的示例中，我們從一台安騰計算機中看到，在gcc版本之間，堆棧的填充和對齊方式發生了很大的變化。這種類型的事情是預期的，並且必須由程序員考慮。


通常，您應該確保不對類型的大小或對齊規則進行假設。


### 帶對齊的習慣用法

有一些處理對齊的普通代碼序列;一般來說，大多數程序會在某些方面考慮它。在處理以某種形式處理數據塊的程序時，您可能會在內核之外的許多地方看到這些“代碼習慣用法”，因此值得研究。


我們可以從Linux內核中獲得一些例子，Linux內核經常需要處理系統內內存頁面的對齊問題。


> 7.3的例子。頁面對齊操作

```c
[ include/asm-ia64/page.h ]

/*
 * PAGE_SHIFT determines the actual kernel page size.
 */
#if defined(CONFIG_IA64_PAGE_SIZE_4KB)
# define PAGE_SHIFT     12
#elif defined(CONFIG_IA64_PAGE_SIZE_8KB)
# define PAGE_SHIFT     13
#elif defined(CONFIG_IA64_PAGE_SIZE_16KB)
# define PAGE_SHIFT     14
#elif defined(CONFIG_IA64_PAGE_SIZE_64KB)
# define PAGE_SHIFT     16
#else
# error Unsupported page size!
#endif

#define PAGE_SIZE               (__IA64_UL_CONST(1) << PAGE_SHIFT)
#define PAGE_MASK               (~(PAGE_SIZE - 1))
#define PAGE_ALIGN(addr)        (((addr) + PAGE_SIZE - 1) & PAGE_MASK)
```


在上面的示例中，我們從一台安騰計算機中看到，在gcc版本之間，堆棧的填充和對齊方式發生了很大的變化。這種類型的事情是預期的，並且必須由程序員考慮。


通常，您應該確保不對類型的大小或對齊規則進行假設。


### 帶對齊的習慣用法

有一些處理對齊的普通代碼序列;一般來說，大多數程序會在某些方面考慮它。在處理以某種形式處理數據塊的程序時，您可能會在內核之外的許多地方看到這些“代碼習慣用法”，因此值得研究。


我們可以從Linux內核中獲得一些例子，Linux內核經常需要處理系統內內存頁面的對齊問題。


在上面，我們可以看到內核中有許多不同的頁面大小選項，範圍從4KB到64KB不等。


PAGE_SIZE宏很容易解釋，通過將值1轉換為給定的移位數(記住，這相當於說2n，其中n是PAGE_SHIFT)，給出系統中選擇的當前頁面大小。


接下來是PAGE_MASK的定義。PAGE_MASK允許我們找到當前頁面中的那些位，即頁面中地址的偏移量。


<!-- XXX繼續簡短討論 -->


### 優化

一旦編譯器有了代碼的內部表示，編譯器真正有趣的部分就開始了。編譯器希望為給定的輸入代碼找到最優化的彙編語言輸出。這是一個大而復雜的問題，需要從基於計算機科學的高效算法到運行代碼的特定處理器的深入知識的所有知識。


在生成輸出時，編譯器可以查看一些常見的優化。有很多很多的策略來生成最好的代碼，並且它總是一個活躍的研究領域。


### 一般優化

編譯器常常可以看到某段代碼不能被使用，因此將它排除在外，將特定的語言結構優化為具有相同結果的更小的結構。頁面對齊操作

### 展開循環

如果代碼包含一個循環(例如for循環或while循環)，並且編譯器知道它將執行多少次，那麼展開循環以便按順序執行可能會更有效。這意味著，與其在循環內部執行，然後分支回開始執行重復的過程，不如復制內部循環代碼，以便再次執行。


雖然這會增加代碼的大小，但它可能允許處理器更有效地處理指令，因為分支可能會導致進入處理器的指令管道的低效。


### 內聯函數

與展開循環類似，可以在被調用函數中插入被調用函數。程序員可以指定編譯器通過在函數定義中內聯指定函數來嘗試這樣做。同樣，通過這樣做，您可以在代碼中按順序交換代碼大小。


### 分支預測

當計算機遇到if語句時，有兩種可能的結果;真或假。處理器希望它的輸入管道盡可能的滿，所以它不能在將代碼放入管道之前等待測試的結果。


因此，編譯器可以預測測試可能進行的方式。編譯器可以使用一些簡單的規則來猜測這樣的事情，例如如果(val == = -1)可能不是真的，因為-1通常表示錯誤代碼，希望不會太頻繁地觸發它。


有些編譯器實際上可以編譯程序，讓用戶運行它，並記錄分支在實際情況下的運行方式。然後，它可以根據所看到的重新編譯它。


> [21] 實際上最常見的形式是擴展 Backus-Naur 形式，或 EBNF，因為它允許一些額外的規則更適合現代語言。


## 彙編


如果您知道處理器的彙編代碼的細節，那麼編譯器輸出的彙編代碼仍然是人類可讀的形式。開發人員通常會查看程序集輸出，以手動檢查代碼是否得到了最優化，或者發現編譯器中的任何錯誤(這比人們可能認為的更常見，特別是當編譯器在進行優化時)。


彙編程序是將彙編代碼轉換成二進制形式的更機械的過程。從本質上講，彙編器保存了一個包含每個可能的指令及其二進制對等物的大表(稱為操作碼的op代碼)。它將這些操作碼與程序集中指定的寄存器結合起來，以生成一個二進制輸出文件。


這個代碼稱為目標代碼，在這個階段，它是不可執行的。目標代碼只是特定輸入源代碼文件的二進制表示形式。良好的編程實踐表明，程序員不應該通過將所有源代碼放在一個文件中來“孤注一擲”。


## 鏈接器

通常在大型程序中，您會將代碼分離到多個文件中，以便將相關函數放在一起。這些文件都可以編譯成目標代碼:但是您的最終目標是創建一個可執行文件!需要以某種方式將這些目標文件組合成單個可執行文件。我們稱之為鏈接。


請注意，即使您的程序確實適合於一個文件，它仍然需要鏈接到某些系統庫才能正確操作。例如，printf調用保存在一個庫中，該庫必須與可執行文件相結合才能工作。因此，雖然在這種情況下您不必顯式地擔心鏈接，但在創建您的可執行文件時，仍然會出現一個鏈接過程。


在下面的部分中，我們將解釋一些理解鏈接所必需的術語。


### 符號

變量和函數在源代碼中都有名稱，我們通過這些名稱引用它們。聲明變量int a的語句的一種方法是告訴編譯器“預留一些sizeof(int)的內存，從現在開始當我使用a它將引用這個分配的內存。”類似地，一個函數說“將此代碼存儲在內存中，當我調用function()時，跳轉到並執行此代碼”。


在本例中，我們將其稱為and函數符號，因為它們是內存區域的符號表示。


符號幫助人們理解編程。可以這樣說，編譯過程的主要工作是刪除符號——處理器不知道a代表什麼，它只知道在特定內存地址有一些數據。編譯過程需要將a += 2轉換為“將內存中的值在0xABCDE上增加2”。

### 符號可見性

在一些C程序中，您可能見過與變量一起使用的術語靜態和extern。這些修飾符可以影響我們所說的符號的可見性。


假設您已經將程序分割為兩個文件，但是有些函數需要共享一個變量。您只需要共享變量的一個定義(即內存位置)(否則它不會被共享!)，但是兩個文件都需要引用它。


為了實現這一點，我們在一個文件中聲明變量，然後在另一個文件中聲明一個名稱相同但前綴為extern的變量。extern代表外在的，對於人類而言，它意味著這個變量被聲明在其他地方。


extern對編譯器說的是，它不應該在內存中為這個變量分配任何空間，而應該將這個符號留在目標代碼中，以供日後修正。編譯器不可能知道符號實際上是在哪裡定義的，但是鏈接器 linker 知道，因為它的工作是將所有對像文件放在一起，並將它們合並到一個可執行文件中。因此鏈接器將看到第二個文件中剩余的符號，並說“我以前在文件1中見過這個符號，我知道它引用內存位置0x12345”。因此，它可以將符號值修改為第一個文件中變量的內存值。


靜態幾乎與extern相反。它對它修改的符號的可見性進行了限制。如果你聲明了一個靜態變量，告訴編譯器“不要在目標代碼中留下任何符號”。這意味著當鏈接器將對像文件鏈接在一起時，它將永遠不會看到那個符號(因此不能使那個“我以前見過這個”連接)。靜態有助於分離和減少衝突——通過聲明變量靜態，您可以在其他文件中重用變量名，而不會導致符號衝突。我們說我們限制了符號的可見性，因為我們不允許鏈接器看到它。與此形成對比的是，鏈接器可以看到一個更可見的符號(一個沒有聲明為static的符號)。


### 鏈接過程

因此鏈接過程實際上是兩個步驟;將所有目標文件組合成一個可執行文件，然後遍歷每個目標文件以解析任何符號。這通常需要兩次傳球;一個是讀取所有符號定義並注意到未解決的符號，另一個是將所有未解決的符號修復到正確的位置。


最終可執行文件應該沒有未解決的符號;如果有鏈接器將失敗並出現錯誤 [22]。


> [22] 我們稱之為靜態鏈接。動態鏈接是在可執行運行時執行的類似概念，稍後將對此進行描述。



### 一個實際的例子

我們可以逐步介紹構建一個簡單應用程序所采取的步驟。


請注意，當您鍵入gcc時，它實際上運行一個驅動程序，該程序對您隱藏了大部分步驟。在正常情況下，這正是您想要的，因為在實際系統上使用可執行文件的命令和選項可能非常復雜，並且是特定於體系結構的。


我們將用下面兩個示例展示編譯過程。這兩個都是C源文件，一個為初始程序入口點定義了main()函數，另一個聲明了一個helper類型函數。還有一個全局變量，僅供舉例。


> 7.4 的例子。 Hello World !

```c
#include <stdio.h>

/* We need a prototype so the compiler knows what types function() takes */
int function(char *input);

/* Since this is static, we can define it in both hello.c and function.c */
static int i = 100;

/* This is a global variable */
int global = 10;

int main(void)
{
	/* function() should return the value of global */
	int ret = function("Hello, World!");
	exit(ret);
}
```

> 7.5的例子。函數的例子

```c
#include <stdio.h>

static int i = 100;

/* Declard as extern since defined in hello.c */
extern int global;

int function(char *input)
{
	printf("%s\n", input);
	return global;
}
```


### 編譯

所有編譯器都可以選擇只執行編譯的第一步。通常這是類似於-S的東西，輸出通常被放入與輸入文件同名但擴展名為.s的文件中。


因此，我們可以使用gcc -S展示第一步，如下例所示。


> 7.6的例子。編譯示例

```
ianw@lime:~/programs/csbu/wk7/code$ gcc -S hello.c
ianw@lime:~/programs/csbu/wk7/code$ gcc -S function.c
ianw@lime:~/programs/csbu/wk7/code$ cat function.s
        .file   "function.c"
        .pred.safe_across_calls p1-p5,p16-p63
        .section        .sdata,"aw",@progbits
        .align 4
        .type   i#, @object
        .size   i#, 4
i:
        data4   100
        .section        .rodata
        .align 8
.LC0:
        stringz "%s\n"
        .text
        .align 16
        .global function#
        .proc function#
function:
        .prologue 14, 33
        .save ar.pfs, r34
        alloc r34 = ar.pfs, 1, 4, 2, 0
        .vframe r35
        mov r35 = r12
        adds r12 = -16, r12
        mov r36 = r1
        .save rp, r33
        mov r33 = b0
        .body
        ;;
        st8 [r35] = r32
        addl r14 = @ltoffx(.LC0), r1
        ;;
        ld8.mov r37 = [r14], .LC0
        ld8 r38 = [r35]
        br.call.sptk.many b0 = printf#
        mov r1 = r36
        ;;
        addl r15 = @ltoffx(global#), r1
        ;;
        ld8.mov r14 = [r15], global#
        ;;
        ld4 r14 = [r14]
        ;;
        mov r8 = r14
        mov ar.pfs = r34
        mov b0 = r33
        .restore sp
        mov r12 = r35
        br.ret.sptk.many b0
        ;;
        .endp function#
        .ident  "GCC: (GNU) 3.3.5 (Debian 1:3.3.5-11)"
```


這個程序集有點復雜，無法完全描述，但是您應該能夠看到i被定義為data4(即4字節或32位，int大小)的位置，其中定義了函數(函數:)和對printf()的調用。


我們現在有兩個文件准備組譯成機器代碼!


### 組譯

組譯是一個相當直接的過程。彙編程序通常被稱為as，並以與gcc類似的方式接受參數


> 7.7的例子。組譯的例子

```
ianw@lime:~/programs/csbu/wk7/code$ as -o function.o function.s
ianw@lime:~/programs/csbu/wk7/code$ as -o hello.o hello.s
ianw@lime:~/programs/csbu/wk7/code$ ls
function.c  function.o  function.s  hello.c  hello.o  hello.s
```


組譯完成後，我們就有了目標代碼，可以鏈接到最終的可執行文件中。通過使用-c調用編譯器，您通常可以跳過手動使用彙編器，編譯器將直接將輸入文件轉換為目標代碼，並將其放入具有相同前綴但擴展名為.o的文件中。


我們不能直接檢查對像代碼，因為它是二進制格式的(在以後的幾周裡，我們將學習這種二進制格式)。但是，我們可以使用一些工具來檢查對像文件，例如readelf——符號將顯示對像文件中的符號。


> 7.8的例子。Readelf例子

```
ianw@lime:~/programs/csbu/wk7/code$ readelf --symbols ./hello.o

Symbol table '.symtab' contains 15 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS hello.c
     2: 0000000000000000     0 SECTION LOCAL  DEFAULT    1
     3: 0000000000000000     0 SECTION LOCAL  DEFAULT    3
     4: 0000000000000000     0 SECTION LOCAL  DEFAULT    4
     5: 0000000000000000     0 SECTION LOCAL  DEFAULT    5
     6: 0000000000000000     4 OBJECT  LOCAL  DEFAULT    5 i
     7: 0000000000000000     0 SECTION LOCAL  DEFAULT    6
     8: 0000000000000000     0 SECTION LOCAL  DEFAULT    7
     9: 0000000000000000     0 SECTION LOCAL  DEFAULT    8
    10: 0000000000000000     0 SECTION LOCAL  DEFAULT   10
    11: 0000000000000004     4 OBJECT  GLOBAL DEFAULT    5 global
    12: 0000000000000000    96 FUNC    GLOBAL DEFAULT    1 main
    13: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND function
    14: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND exit

ianw@lime:~/programs/csbu/wk7/code$ readelf --symbols ./function.o

Symbol table '.symtab' contains 14 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS function.c
     2: 0000000000000000     0 SECTION LOCAL  DEFAULT    1
     3: 0000000000000000     0 SECTION LOCAL  DEFAULT    3
     4: 0000000000000000     0 SECTION LOCAL  DEFAULT    4
     5: 0000000000000000     0 SECTION LOCAL  DEFAULT    5
     6: 0000000000000000     4 OBJECT  LOCAL  DEFAULT    5 i
     7: 0000000000000000     0 SECTION LOCAL  DEFAULT    6
     8: 0000000000000000     0 SECTION LOCAL  DEFAULT    7
     9: 0000000000000000     0 SECTION LOCAL  DEFAULT    8
    10: 0000000000000000     0 SECTION LOCAL  DEFAULT   10
    11: 0000000000000000   128 FUNC    GLOBAL DEFAULT    1 function
    12: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND printf
    13: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND global
```


盡管輸出相當復雜(又一次!)，但您應該能夠理解其中的大部分內容。例如


在hello的輸出中。看看這個名字為i的符號，注意它是如何表示本地的嗎?這是因為我們聲明它是靜態的，因此它被標記為這個對像文件的本地文件。


在相同的輸出中，請注意全局變量被定義為全局變量，這意味著它在這個文件之外可見。同樣，main()函數在外部可見。


注意，函數符號(用於對函數()的調用)左有一個UND或undefined。這意味著它被留給鏈接器去查找函數的地址。


看看函數中的符號。c文件，以及它們如何適合輸出。


### 鏈接

實際上，調用名為ld的鏈接器在實際系統上是一個非常復雜的過程(您已經聽膩了嗎?)這就是我們將鏈接過程留給gcc的原因。


但是，當然，我們可以通過-v(詳細)標志來監視gcc在幕後的活動。


> 7.9的例子。鏈接的例子

```
/usr/lib/gcc-lib/ia64-linux/3.3.5/collect2 -static 
/usr/lib/gcc-lib/ia64-linux/3.3.5/../../../crt1.o 
/usr/lib/gcc-lib/ia64-linux/3.3.5/../../../crti.o 
/usr/lib/gcc-lib/ia64-linux/3.3.5/crtbegin.o 
-L/usr/lib/gcc-lib/ia64-linux/3.3.5 
-L/usr/lib/gcc-lib/ia64-linux/3.3.5/../../.. 
hello.o 
function.o 
--start-group 
-lgcc 
-lgcc_eh 
-lunwind 
-lc 
--end-group 
/usr/lib/gcc-lib/ia64-linux/3.3.5/crtend.o 
/usr/lib/gcc-lib/ia64-linux/3.3.5/../../../crtn.o
```


首先您會注意到一個名為collect2的程序正在被調用。這是gcc內部使用的一個簡單的ld包裝器。


您注意到的下一件事是對像文件，從指定鏈接器的crt開始。這些函數由gcc和系統庫提供，包含啟動程序所需的代碼。實際上，main()函數不是程序運行時調用的第一個函數，而是crt對像文件中名為_start的函數。這個函數執行一些應用程序程序員不需要擔心的通用設置。


路徑層次結構相當復雜，但從本質上講，我們可以看到最後一步是鏈接一些額外的對像文件，即


* crt1.o:由系統庫(libc)提供的這個對像文件包含_start函數，它實際上是程序中調用的第一個函數。
  crti.o : provided by the system libraries
  crtbegin.o
  crtsaveres.o
  crtend.o
  crtn.o

首先您會注意到一個名為collect2的程序正在被調用。這是gcc內部使用的一個簡單的ld包裝器。


您注意到的下一件事是對像文件，從指定鏈接器的crt開始。這些函數由gcc和系統庫提供，包含啟動程序所需的代碼。實際上，main()函數不是程序運行時調用的第一個函數，而是crt對像文件中名為_start的函數。這個函數執行一些應用程序程序員不需要擔心的通用設置。


路徑層次結構相當復雜，但從本質上講，我們可以看到最後一步是鏈接一些額外的對像文件，即


* crt1。我們稍後將討論如何使用它們來啟動程序。


接下來您可以看到我們在兩個對像文件中鏈接，hello。o和function.o。然後，我們用-l標志指定一些額外的庫。這些庫是系統特有的，每個程序都需要。主要的一個是-lc，它引入了C庫，它具有所有常見的函數，如printf()。


之後，我們再次鏈接到更多的系統對像文件中，這些文件在程序退出後進行一些清理。


雖然細節很復雜，但概念是直截了當的。所有的目標文件將被鏈接到一個單一的可執行文件，准備運行!


### 可執行文件

在不久的將來，我們將深入討論可執行文件的更多細節，但是我們可以以類似於對像文件的方式進行一些檢查，看看發生了什麼。


> 7.10的例子。可執行文件的例子

```
ianw@lime:~/programs/csbu/wk7/code$ gcc -o program hello.c function.c
ianw@lime:~/programs/csbu/wk7/code$ readelf --symbols ./program

Symbol table '.dynsym' contains 11 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 6000000000000de0     0 OBJECT  GLOBAL DEFAULT  ABS _DYNAMIC
     2: 0000000000000000   176 FUNC    GLOBAL DEFAULT  UND printf@GLIBC_2.2 (2)
     3: 600000000000109c     0 NOTYPE  GLOBAL DEFAULT  ABS __bss_start
     4: 0000000000000000   704 FUNC    GLOBAL DEFAULT  UND exit@GLIBC_2.2 (2)
     5: 600000000000109c     0 NOTYPE  GLOBAL DEFAULT  ABS _edata
     6: 6000000000000fe8     0 OBJECT  GLOBAL DEFAULT  ABS _GLOBAL_OFFSET_TABLE_     7: 60000000000010b0     0 NOTYPE  GLOBAL DEFAULT  ABS _end
     8: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _Jv_RegisterClasses
     9: 0000000000000000   544 FUNC    GLOBAL DEFAULT  UND __libc_start_main@GLIBC_2.2 (2)
    10: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND __gmon_start__

Symbol table '.symtab' contains 127 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 40000000000001c8     0 SECTION LOCAL  DEFAULT    1
     2: 40000000000001e0     0 SECTION LOCAL  DEFAULT    2
     3: 4000000000000200     0 SECTION LOCAL  DEFAULT    3
     4: 4000000000000240     0 SECTION LOCAL  DEFAULT    4
     5: 4000000000000348     0 SECTION LOCAL  DEFAULT    5
     6: 40000000000003d8     0 SECTION LOCAL  DEFAULT    6
     7: 40000000000003f0     0 SECTION LOCAL  DEFAULT    7
     8: 4000000000000410     0 SECTION LOCAL  DEFAULT    8
     9: 4000000000000440     0 SECTION LOCAL  DEFAULT    9
    10: 40000000000004a0     0 SECTION LOCAL  DEFAULT   10
    11: 40000000000004e0     0 SECTION LOCAL  DEFAULT   11
    12: 40000000000005e0     0 SECTION LOCAL  DEFAULT   12
    13: 4000000000000b00     0 SECTION LOCAL  DEFAULT   13
    14: 4000000000000b40     0 SECTION LOCAL  DEFAULT   14
    15: 4000000000000b60     0 SECTION LOCAL  DEFAULT   15
    16: 4000000000000bd0     0 SECTION LOCAL  DEFAULT   16
    17: 4000000000000ce0     0 SECTION LOCAL  DEFAULT   17
    18: 6000000000000db8     0 SECTION LOCAL  DEFAULT   18
    19: 6000000000000dd0     0 SECTION LOCAL  DEFAULT   19
    20: 6000000000000dd8     0 SECTION LOCAL  DEFAULT   20
    21: 6000000000000de0     0 SECTION LOCAL  DEFAULT   21
    22: 6000000000000fc0     0 SECTION LOCAL  DEFAULT   22
    23: 6000000000000fd0     0 SECTION LOCAL  DEFAULT   23
    24: 6000000000000fe0     0 SECTION LOCAL  DEFAULT   24
    25: 6000000000000fe8     0 SECTION LOCAL  DEFAULT   25
    26: 6000000000001040     0 SECTION LOCAL  DEFAULT   26
    27: 6000000000001080     0 SECTION LOCAL  DEFAULT   27
    28: 60000000000010a0     0 SECTION LOCAL  DEFAULT   28
    29: 60000000000010a8     0 SECTION LOCAL  DEFAULT   29
    30: 0000000000000000     0 SECTION LOCAL  DEFAULT   30
    31: 0000000000000000     0 SECTION LOCAL  DEFAULT   31
    32: 0000000000000000     0 SECTION LOCAL  DEFAULT   32
    33: 0000000000000000     0 SECTION LOCAL  DEFAULT   33
    34: 0000000000000000     0 SECTION LOCAL  DEFAULT   34
    35: 0000000000000000     0 SECTION LOCAL  DEFAULT   35
    36: 0000000000000000     0 SECTION LOCAL  DEFAULT   36
    37: 0000000000000000     0 SECTION LOCAL  DEFAULT   37
    38: 0000000000000000     0 SECTION LOCAL  DEFAULT   38
    39: 0000000000000000     0 SECTION LOCAL  DEFAULT   39
    40: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    41: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    42: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    43: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    44: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    45: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    46: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    47: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    48: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    49: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <built-in>
    50: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS abi-note.S
    51: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    52: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS abi-note.S
    53: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    54: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS abi-note.S
    55: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    56: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    57: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    58: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <built-in>
    59: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS abi-note.S
    60: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS init.c
    61: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    62: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    63: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS initfini.c
    64: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    65: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    66: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    67: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    68: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <built-in>
    69: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    70: 4000000000000670   128 FUNC    LOCAL  DEFAULT   12 gmon_initializer
    71: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    72: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    73: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS initfini.c
    74: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    75: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    76: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    77: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    78: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <built-in>
    79: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS /build/buildd/glibc-2.3.2
    80: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS auto-host.h
    81: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    82: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <built-in>
    83: 6000000000000fc0     0 NOTYPE  LOCAL  DEFAULT   22 __CTOR_LIST__
    84: 6000000000000fd0     0 NOTYPE  LOCAL  DEFAULT   23 __DTOR_LIST__
    85: 6000000000000fe0     0 NOTYPE  LOCAL  DEFAULT   24 __JCR_LIST__
    86: 6000000000001088     8 OBJECT  LOCAL  DEFAULT   27 dtor_ptr
    87: 40000000000006f0   128 FUNC    LOCAL  DEFAULT   12 __do_global_dtors_aux    
    88: 4000000000000770   128 FUNC    LOCAL  DEFAULT   12 __do_jv_register_classes
    89: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS hello.c
    90: 6000000000001090     4 OBJECT  LOCAL  DEFAULT   27 i
    91: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS function.c
    92: 6000000000001098     4 OBJECT  LOCAL  DEFAULT   27 i
    93: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS auto-host.h
    94: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <command line>
    95: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS <built-in>
    96: 6000000000000fc8     0 NOTYPE  LOCAL  DEFAULT   22 __CTOR_END__
    97: 6000000000000fd8     0 NOTYPE  LOCAL  DEFAULT   23 __DTOR_END__
    98: 6000000000000fe0     0 NOTYPE  LOCAL  DEFAULT   24 __JCR_END__
    99: 6000000000000de0     0 OBJECT  GLOBAL DEFAULT  ABS _DYNAMIC
   100: 4000000000000a70   144 FUNC    GLOBAL HIDDEN   12 __do_global_ctors_aux
   101: 6000000000000dd8     0 NOTYPE  GLOBAL DEFAULT  ABS __fini_array_end
   102: 60000000000010a8     8 OBJECT  GLOBAL HIDDEN   29 __dso_handle
   103: 40000000000009a0   208 FUNC    GLOBAL DEFAULT   12 __libc_csu_fini
   104: 0000000000000000   176 FUNC    GLOBAL DEFAULT  UND printf@@GLIBC_2.2
   105: 40000000000004a0    32 FUNC    GLOBAL DEFAULT   10 _init
   106: 4000000000000850   128 FUNC    GLOBAL DEFAULT   12 function
   107: 40000000000005e0   144 FUNC    GLOBAL DEFAULT   12 _start
   108: 6000000000001094     4 OBJECT  GLOBAL DEFAULT   27 global
   109: 6000000000000dd0     0 NOTYPE  GLOBAL DEFAULT  ABS __fini_array_start
   110: 40000000000008d0   208 FUNC    GLOBAL DEFAULT   12 __libc_csu_init
   111: 600000000000109c     0 NOTYPE  GLOBAL DEFAULT  ABS __bss_start
   112: 40000000000007f0    96 FUNC    GLOBAL DEFAULT   12 main
   113: 6000000000000dd0     0 NOTYPE  GLOBAL DEFAULT  ABS __init_array_end
   114: 6000000000000dd8     0 NOTYPE  WEAK   DEFAULT   20 data_start
   115: 4000000000000b00    32 FUNC    GLOBAL DEFAULT   13 _fini
   116: 0000000000000000   704 FUNC    GLOBAL DEFAULT  UND exit@@GLIBC_2.2
   117: 600000000000109c     0 NOTYPE  GLOBAL DEFAULT  ABS _edata
   118: 6000000000000fe8     0 OBJECT  GLOBAL DEFAULT  ABS _GLOBAL_OFFSET_TABLE_   
   119: 60000000000010b0     0 NOTYPE  GLOBAL DEFAULT  ABS _end
   120: 6000000000000db8     0 NOTYPE  GLOBAL DEFAULT  ABS __init_array_start
   121: 6000000000001080     4 OBJECT  GLOBAL DEFAULT   27 _IO_stdin_used
   122: 60000000000010a0     8 OBJECT  GLOBAL DEFAULT   28 __libc_ia64_register_back
   123: 6000000000000dd8     0 NOTYPE  GLOBAL DEFAULT   20 __data_start
   124: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _Jv_RegisterClasses
   125: 0000000000000000   544 FUNC    GLOBAL DEFAULT  UND __libc_start_main@@GLIBC_
   126: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND __gmon_start__
```

有幾點需要注意


* 注意我用“簡單”的方式構建了可執行文件!
* 見有兩個符號表;dynsym和symtab。我們將很快解釋dynsym符號是如何工作的，但是請注意，其中一些符號是用@符號進行版本化的。
* 注意從額外的目標文件中包含的許多符號。其中許多以__開頭，以避免與程序員可能選擇的任何名稱衝突。通讀並從目標文件中挑出我們之前提到的符號，看看它們是否有任何變化。