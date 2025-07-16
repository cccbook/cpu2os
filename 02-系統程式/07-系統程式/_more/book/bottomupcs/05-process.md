<!--

#Chapter 5. The Process

##What is a process?

We are all familiar with the modern operating system running many tasks all at once or multitasking.

We can think of each process as a bundle of elements kept by the kernel to keep track of all these running tasks.

##Elements of a process

> Figure 5.1. The Elements of a Process

![](http://www.bottomupcs.com/chapter04/figures/theprocess.png)

The essential elements of a process; the process ID, memory, files and registers.
----

###Process ID
The process ID (or the PID) is assigned by the operating system and is unique to each running process.

###Memory
We will learn exactly how a process gets its memory in the following weeks -- it is one of the most fundamental parts of how the operating system works. However, for now it is sufficient to know that each process gets its own section of memory.

In this memory all the program code is stored, along with variables and any other allocated storage.

Parts of the memory can be shared between processes (called, not surprisingly shared memory). You will often see this called System Five Shared Memory (or SysV SHM) after the original implementation in an older operating system.

Another important concept a process may utilise is that of mmaping a file on disk to memory. This means that instead of having to open the file and use commands such as read() and write() the file looks as if it were any other type of RAM. mmaped areas have permissions such as read, write and execute which need to be kept track of. As we know, it is the job of the operating system to maintain security and stability, so it needs to check if a process tries to write to a read only area and return an error.

###Code and Data
A process can be further divided into code and data sections. Program code and data should be kept separately since they require different permissions from the operating system and separation facilitates sharing of code (as you see later). The operating system needs to give program code permission to be read and executed, but generally not written to. On the other hand data (variables) require read and write permissions but should not be executable[12].
-->

# 第五章。 進程 Process


## 什麼是進程?


我們都熟悉現代操作系統，它可以同時運行許多任務或多任務。


我們可以將每個進程看作是內核保存的一組元素，以跟蹤所有這些正在運行的任務。


## 進程的元素


>圖5.1。進程的元素


![](http://www.bottomupcs.com/chapter04/figures/theprocess.png)


> 進程的基本要素;進程ID、內存、文件和寄存器。



### 進程ID

進程ID(或PID)由操作系統分配，對於每個運行的進程是唯一的。


### 內存

我們將在接下來的幾週裡確切地瞭解進程是如何獲得內存的——它是操作系統工作方式中最基本的部分之一。然而，現在只要知道每個進程都有自己的內存部分就足夠了。


在這個內存中，所有程序代碼和變數以及任何其他分配的存儲一起存儲。


內存的部分可以在進程之間共享(稱為共享內存，這並不奇怪)。在舊的操作系統中最初實現之後，您將經常看到這個稱為System 5共享內存(或SysV SHM)的系統。


進程可能使用的另一個重要概念是將磁碟上的文件映射到內存中。這意味著，與打開文件並使用read()和write()等命令不同，文件看起來像是其他類型的RAM。mmaped區域具有讀、寫和執行等權限，需要跟蹤這些權限。正如我們所知，操作系統的工作是維護安全性和穩定性，因此它需要檢查進程是否試圖寫入只讀區域並返回錯誤。


### 代碼和數據

流程可以進一步分為代碼和數據部分。程序代碼和數據應該分開保存，因為它們需要與操作系統不同的權限，分離有助於代碼的共享(如您稍後所見)。操作系統需要給程序代碼權限才能被讀取和執行，但通常不會被寫入。另一方面，數據(變數)需要讀寫權限，但不應該是可執行的[12]。

<!--

###The Stack
One other very important part of a process is an area of memory called the stack. This can be considered part of the data section of a process, and is intimately involved in the execution of any program.

A stack is generic data structure that works exactly like a stack of plates; you can push an item (put a plate on top of a stack of plates), which then becomes the top item, or you can pop an item (take a plate off, exposing the previous plate).

Stacks are fundamental to function calls. Each time a function is called it gets a new stack frame. This is an area of memory which usually contains, at a minimum, the address to return to when complete, the input arguments to the function and space for local variables.

By convention, stacks usually grow down[13] . This means that the stack starts at a high address in memory and progressively gets lower.

Figure 5.2. The Stack

![](http://www.bottomupcs.com/chapter04/figures/stack.png)

How the stack works with function calls. Note that the stack grows downwards, from high addresses in memory to low addresses.

We can see how having a stack brings about many of the features of functions.

* Each function has its own copy of its input arguments. This is because each function is allocated a new stack frame with its arguments in a fresh area of memory.
* This is the reason why a variable defined inside a function can not be seen by other functions. Global variables (which can be seen by any function) are kept in a separate area of data memory.
* This facilitates recursive calls. This means a function is free to call itself again, because a new stack frame will be created for all its local variables.
* Each frame contains the address to return to. C only allows a single value to be returned from a function, so by convention this value is returned to the calling function in a specified register, rather than on the stack.
* Because each frame has a reference to the one before it, a debugger can "walk" backwards, following the pointers up the stack. From this it can produce a stack trace which shows you all functions that were called leading into this function. This is extremely useful for debugging.
* You can see how the way functions works fits exactly into the nature of a stack. Any function can call any other function, which then becomes the up most function (put on top of the stack). Eventually that function will return to the function that called it (takes itself off the stack).
* Stacks do make calling functions slower, because values must be moved out of registers and into memory. Some architectures allow arguments to be passed in registers directly; however to keep the semantics that each function gets a unique copy of each argument the registers must rotate.
* You may have heard of the term a stack overflow. This is a common way of hacking a system by passing bogus values. If you as a programmer accept arbitrary input into a stack variable (say, reading from the keyboard or over the network) you need to explicitly say how big that data is going to be.
* Allowing any amount of data unchecked will simply overwrite memory. Generally this leads to a crash, but some people realised that if they overwrote just enough memory to place a specific value in the return address part of the stack frame, when the function completed rather than returning to the correct place (where it was called from) they could make it return into the data they just sent. If that data contains binary executable code that hacks the system (e.g. starts a terminal for the user with root privileges) then your computer has been compromised.
* This happens because the stack grows downwards, but data is read in "upwards" (i.e. from lower address to higher addresses).
* There are several ways around this; firstly as a programmer you must ensure that you always check the amount of data you are receiving into a variable. The operating system can help to avoid this on behalf of the programmer by ensuring that the stack is marked as not executable; that is that the processor will not run any code, even if a malicious user tries to pass some into your program. Modern architectures and operating systems support this functionality.
* Stacks are ultimately managed by the compiler, as it is responsible for generating the program code. To the operating system the stack just looks like any other area of memory for the process.
* To keep track of the current growth of the stack, the hardware defines a register as the stack pointer. The compiler (or the programmer, when writing in assembler) uses this register to keep track of the current top of the stack.

> Example 5.1. Stack pointer example
-->
### 堆棧

進程的另一個非常重要的部分是一個被稱為堆棧的內存區域。這可以看作是流程數據部分的一部分，並且與任何程序的執行密切相關。


棧是一種通用的數據結構，它的工作原理與一堆板材完全一樣;你可以推一個物品(把一個盤子放在一堆盤子的上面)，它就變成了最上面的物品，或者你也可以把一個物品(把盤子拿下來，露出上面的盤子)。


堆棧是函數調用的基礎。每次調用一個函數都會得到一個新的堆棧框架。這是一個內存區域，通常至少包含完成時要返回的地址、函數的輸入參數和局部變數的空間。


按照慣例，棧通常會增長到[13]以下。這意味著堆棧從內存中的高地址開始，然後逐漸降低。


> 圖5.2。堆棧


![](http://www.bottomupcs.com/chapter04/figures/stack.png)


> 堆棧如何與函數調用一起工作。注意，堆棧是向下增長的，從內存中的高地址增長到低地址。


我們可以看到堆棧如何帶來函數的許多特性。


* 每個函數都有自己的輸入參數副本。這是因為每個函數都被分配了一個新的堆棧框架，它的參數位於一個新的內存區域中。這就是為什麼函數內部定義的變數不能被其他函數看到的原因。全局變數(任何函數都可以看到)保存在數據內存的單獨區域中。
* 這有助於遞歸調用。這意味著函數可以自由地再次調用自身，因為將為其所有局部變數創建一個新的堆棧框架。
* 每一幀包含要返回的地址。C只允許從函數返回一個值，因此按照約定，這個值將在指定的寄存器中返回給調用函數，而不是在堆棧中。
* 因為每個幀都有一個對它前面的幀的引用，所以調試器可以“向後走”，跟隨堆棧上的指針。從這裡，它可以生成一個堆棧跟蹤，該跟蹤向您顯示所有調用的引導到此函數的函數。這對於調試非常有用。
* 您可以看到函數的工作方式完全符合堆棧的本質。任何函數都可以調用任何其他函數，這就變成了最上面的函數(放在堆棧的頂部)。最終，該函數將返回到調用它的函數(從堆棧中取出自己)。
  堆棧確實使調用函數變慢，因為值必須從寄存器移到內存中。一些體系結構允許參數直接在寄存器中傳遞;然而，為了保持每個函數獲得每個參數的唯一副本的語義，寄存器必須旋轉。
* 您可能聽說過棧溢出這個術語。這是通過傳遞假值來攻擊系統的一種常見方式。如果您作為一名程序員接受對堆棧變數的任意輸入(例如，從鍵盤或網絡上讀取)，您需要明確地說明數據的大小。
* 允許任意數量的數據不受檢查只會覆蓋內存。通常這會導致崩潰,但有些人意識到,如果他們只覆蓋了足夠的內存中的一個特定的值返回地址棧框架的一部分,當函數完成而不是回到正確的位置(從)他們能返回到數據發送。如果該數據包含二進制可執行代碼，會破壞系統(例如，為用戶啟動具有root權限的終端)，那麼您的計算機就被破壞了。
* 這是因為堆棧向下增長，但數據是“向上”讀取(即從低地址到高地址)。
	有幾種方法可以解決這個問題;首先，作為一名程序員，您必須確保始終將接收到的數據量檢查到一個變數中。操作系統可以確保堆棧被標記為不可執行的，從而幫助程序員避免這種情況;也就是說，處理器不會運行任何代碼，即使惡意用戶試圖向您的程序傳遞一些代碼。現代體系結構和操作系統支持這種功能。

	棧最終由編譯器管理，因為它負責生成程序代碼。對於操作系統來說，堆棧看起來就像進程的其他內存區域。
* 為了跟蹤堆棧的當前增長，硬件將寄存器定義為堆棧指針。編譯器(或程序員，當用彙編程序編寫時)使用這個寄存器來跟蹤堆棧的當前頂部。


> 5.1的例子。堆棧指針的例子

```
$ cat sp.c
void function(void)
{
        int i = 100;
        int j = 200;
        int k = 300;
}

$ gcc -fomit-frame-pointer -S sp.c

$ cat sp.s
        .file   "sp.c"
        .text
.globl function
        .type   function, @function
function:
        subl    $16, %esp
        movl    $100, 4(%esp)
        movl    $200, 8(%esp)
        movl    $300, 12(%esp)
        addl    $16, %esp
        ret
        .size   function, .-function
        .ident  "GCC: (GNU) 4.0.2 20050806 (prerelease) (Debian 4.0.1-4)"
        .section        .note.GNU-stack,"",@progbits
```

<!--

Above we show a simple function allocating three variables on the stack. The disassembly illustrates the use of the stack pointer on the x86 architecture[14]. Firstly we allocate some space on the stack for our local variables. Since the stack grows down, we subtract from the value held in the stack pointer. The value 16 is a value large enough to hold our local variables, but may not be exactly the size required (for example with 3 4 byte int values we really only need 12 bytes, not 16) to keep alignment of the stack in memory on certain boundaries as the compiler requires.

Then we move the values into the stack memory (and in a real function, use them). Finally, before returning to our parent function we "pop" the values off the stack by moving the stack pointer back to where it was before we started.

###The Heap
The heap is an area of memory that is managed by the process for on the fly memory allocation. This is for variables whose memory requirements are not known at compile time.

The bottom of the heap is known as the brk, so called for the system call which modifies it. By using the brk call to grow the area downwards the process can request the kernel allocate more memory for it to use.

The heap is most commonly managed by the malloc library call. This makes managing the heap easy for the programmer by allowing them to simply allocate and free (via the free call) heap memory. malloc can use schemes like a buddy allocator to manage the heap memory for the user. malloc can also be smarter about allocation and potentially use anonymous mmaps for extra process memory. This is where instead of mmaping a file into the process memory it directly maps an area of system RAM. This can be more efficient. Due to the complexity of managing memory correctly, it is very uncommon for any modern program to have a reason to call brk directly.

###Memory Layout

Figure 5.3. Process memory layout
-->
上面我們展示了一個在堆棧上分配三個變數的簡單函數。拆卸演示了x86體系結構[14]上堆棧指針的使用。首先，我們在堆棧上為局部變數分配一些空間。由於堆棧在減少，我們從堆棧指針中保存的值中減去。16是一個值的價值足夠容納我們的本地變數,但可能不是完全尺寸的要求(例如,3 4位元組int值我們真的只需要12位元組,不是16)保持對齊內存堆棧的某些編譯器所需要的邊界。


然後我們將這些值移動到堆棧內存中(在實函數中，使用它們)。最後，在返回到父函數之前，我們通過將堆棧指針移回開始前的位置來“彈出”堆棧中的值。


### 堆

堆是由進程管理的內存區域，用於動態內存分配。這是針對那些在編譯時不知道內存需求的變數。


堆的底部被稱為brk，即修改它的系統調用。通過使用brk調用向下擴展區域，進程可以請求內核分配更多內存供其使用。


堆通常由malloc庫調用管理。這使得程序員可以通過簡單地分配和釋放堆內存(通過自由調用)來輕鬆地管理堆。malloc可以使用夥伴分配器之類的方案來管理用戶的堆內存。malloc還可以更聰明地進行分配，並可能使用匿名映射來獲得額外的進程內存。在這裡，它不是將文件映射到進程內存中，而是直接映射系統RAM的區域。這樣會更有效率。由於正確管理內存的複雜性，任何現代程序都沒有理由直接調用brk。


### 內存佈局


圖5.3。進程的內存佈局

![](http://www.bottomupcs.com/chapter04/figures/memory-layout.png)

<!--

> A sample of how process memory is allocated.

As we have seen a process has smaller areas of memory allocated to it, each with a specific purpose.

An example of how the process is laid out in memory by the kernel is given above. Starting from the top, the kernel reserves itself some memory at the top of the process (we see with virtual memory how this memory is actually shared between all processes).

Underneath that is room for mmaped files and libraries. Underneath that is the stack, and below that the heap.

At the bottom is the program image, as loaded from the executable file on disk. We take a closer look at the process of loading this data in later chapters.

###File Descriptors
In the first week we learnt about stdin, stdout and stderr; the default files given to each process. You will remember that these files always have the same file descriptor number (0,1,2 respectively).

Thus, file descriptors are kept by the kernel individually for each process.

File descriptors also have permissions. For example, you may be able to read from a file but not write to it. When the file is opened, the operating system keeps a record of the processes permissions to that file in the file descriptor and doesn't allow the process to do anything it shouldn't.

###Registers
We know from the previous chapter that the processor essentially performs generally simple operations on values in registers. These values are read (and written) to memory -- we mentioned above that each process is allocated memory which the kernel keeps track of.

So the other side of the equation is keeping track of the registers. When it comes time for the currently running process to give up the processor so another process can run, it needs to save its current state. Equally, we need to be able to restore this state when the process is given more time to run on the CPU. To do this the operating system needs to store a copy of the CPU registers to memory. When it is time for the process to run again, the operating system will copy the register values back from memory to the CPU registers and the process will be right back where it left off.

###Kernel State
Internally, the kernel needs to keep track of a number of elements for each process.

###Process State
Another important element for the operating system to keep track of is the process state. If the process is currently running it makes sense to have it in a running state.

However, if the process has requested to read a file from disk we know from our memory hierarchy that this may take a significant amount of time. The process should give up its current execution to allow another process to run, but the kernel need not let the process run again until the data from the disk is available in memory. Thus it can mark the process as disk wait (or similar) until the data is ready.

###Priority
Some processes are more important than others, and get a higher priority. See the discussion on the scheduler below.

###Statistics

The kernel can keep statistics on each processes behaviour which can help it make decisions about how the process behaves; for example does it mostly read from disk or does it mostly do CPU intensive operations?
-->
> 進程內存的分配示例。


正如我們所看到的，一個進程有更小的內存分配區域，每個區域都有特定的用途。


上面給出了一個由內核在內存中佈局進程的例子。從最上面開始，內核會在進程的頂部保留一些內存(通過虛擬內存，我們可以看到這個內存是如何在所有進程之間共享的)。


下面是mmaped文件和庫的空間。下面是堆棧，下面是堆。


底部是程序映像，從磁碟上的可執行文件加載。在後面的章節中，我們將更深入地瞭解加載這些數據的過程。


### 文件描述符

第一週我們瞭解了stdin, stdout和stderr;給每個進程的默認文件。您將記住，這些文件總是具有相同的文件描述符號(分別為0,1,2)。


因此，內核為每個進程單獨保存文件描述符。


文件描述符也有權限。例如，您可以從文件中讀取，但不能寫入。當文件被打開時，操作系統在文件描述符中保存對該文件的進程權限的記錄，不允許進程做任何不應該做的事情。


### 寄存器

從上一章我們知道，處理器基本上對寄存器中的值執行簡單的操作。這些值被讀(和寫)到內存中——我們在上面提到過，每個進程都被分配給內核跟蹤的內存。


方程的另一邊是記錄寄存器。噹噹前正在運行的進程放棄處理器以便另一個進程可以運行時，它需要保存當前狀態。同樣，當進程有更多時間在CPU上運行時，我們需要能夠恢復這種狀態。為此，操作系統需要將CPU寄存器的副本存儲到內存中。當該進程再次運行時，操作系統將把寄存器值從內存複製回CPU寄存器，進程將回到它停止的地方。


### 內核狀態

在內部，內核需要跟蹤每個進程的許多元素。


### 流程狀態

操作系統要跟蹤的另一個重要元素是進程狀態。如果進程當前正在運行，那麼讓它處於運行狀態是有意義的。


然而，如果進程請求從磁碟讀取文件，我們從內存層次結構中知道這可能會花費大量時間。進程應該放棄當前的執行以允許另一個進程運行，但是內核不需要讓進程再次運行，直到磁碟中的數據在內存中可用。因此，它可以將進程標記為磁碟等待(或類似)，直到數據就緒。


### 優先

有些進程比其他進程更重要，並且獲得更高的優先級。參見下面關於調度程序的討論。


### 統計


內核可以對每個進程的行為進行統計，從而幫助它決定進程的行為;例如，它主要從磁碟讀取還是主要執行CPU密集型操作?

<!--

> [12] Not all architectures support this, however. This has lead to a wide range of security problems on many architectures.
> 
> [13] Some architectures, such as PA-RISC from HP, have stacks that grow upwards. On some other architectures, such as IA64, there are other storage areas (the register backing store) that grow from the bottom toward the stack.> 
> 
> [14] Note we used the special flag to gcc -fomit-frame-pointer which specifies that an extra register should not be used to keep a pointer to the start of the stack frame. Having this pointer helps debuggers to walk upwards through the stack frames, however it makes one less register available for other applications.

##Process Hierarchy

Whilst the operating system can run many processes at the same time, in fact it only ever directly starts one process called the init (short for initial) process. This isn't a particularly special process except that its PID is always 0 and it will always be running.

All other processes can be considered children of this initial process. Processes have a family tree just like any other; each process has a parent and can have many siblings, which are processes created[15] by the same parent.

Certainly children can create more children and so on and so forth.

> Example 5.2. pstree example
-->
然而，並不是所有的體系結構都支持這一點。這在許多體系結構上導致了廣泛的安全問題。

>

一些架構，例如HP的PA-RISC，有向上增長的棧。在其他一些體系結構(如IA64)上，還有其他存儲區域(寄存器後備存儲)從底部向堆棧增長。>

>

注意，我們使用特殊標誌來表示gcc -fomit-frame-pointer，它指定不應該使用額外的寄存器來保持指向堆棧框架開始的指針。有了這個指針，調試器就可以向上遍歷堆棧幀，但是它減少了一個寄存器對其他應用程序的可用性。


## 進程層次結構


雖然操作系統可以同時運行多個進程，但實際上它只直接啟動一個名為init(初始進程的簡稱)的進程。這不是一個特別特別的進程，除了它的PID總是0並且它總是在運行。


所有其他進程都可以看作是這個初始進程的子進程。進程和其他進程一樣有一個家譜;每個進程都有一個父進程，並且可以有多個兄弟進程，它們是由同一個父進程創建的[15]進程。


當然孩子可以創造更多的孩子等等。


> 5.2的例子。pstree例子

```
init-+-apmd
     |-atd
     |-cron
     ...
     |-dhclient
     |-firefox-bin-+-firefox-bin---2*[firefox-bin]
     |             |-java_vm---java_vm---13*[java_vm]
     |             `-swf_play
```

<!--

> [15] The term spawn is often used when talking about parent processes creating children; as in "the process spawned a child".

##Fork and Exec
New processes are created by the two related interfaces fork and exec.

###Fork
When you come to metaphorical "fork in the road" you generally have two options to take, and your decision effects your future. Computer programs reach this fork in the road when they hit the fork() system call.

At this point, the operating system will create a new process that is exactly the same as the parent process. This means all the state that was talked about previously is copied, including open files, register state and all memory allocations, which includes the program code.

The return value from the system call is the only way the process can determine if it was the existing process or a new one. The return value to the parent process will be the Process ID (PID) of the child, whilst the child will get a return value of 0.

At this point, we say the process has forked and we have the parent-child relationship as described above.

###Exec
Forking provides a way for an existing process to start a new one, but what about the case where the new process is not part of the same program as parent process? This is the case in the shell; when a user starts a command it needs to run in a new process, but it is unrelated to the shell.

This is where the exec system call comes into play. exec will replace the contents of the currently running process with the information from a program binary.

Thus the process the shell follows when launching a new program is to firstly fork, creating a new process, and then exec (i.e. load into memory and execute) the program binary it is supposed to run.

###How Linux actually handles fork and exec

####clone
In the kernel, fork is actually implemented by a clone system call. This clone interfaces effectively provides a level of abstraction in how the Linux kernel can create processes.

clone allows you to explicitly specify which parts of the new process are copied into the new process, and which parts are shared between the two processes. This may seem a bit strange at first, but allows us to easily implement threads with one very simple interface.

####Threads
While fork copies all of the attributes we mentioned above, imagine if everything was copied for the new process except for the memory. This means the parent and child share the same memory, which includes program code and data.

> Figure 5.4. Threads
-->
當談到父進程創建子進程時，通常使用衍生這個術語;就像“這個進程孕育了一個孩子”。


### Fork和Exec

新的進程由fork和exec這兩個相關接口創建。


### 叉

當你遇到比喻意義上的“岔路口”時，你通常有兩種選擇，而你的決定會影響你的未來。計算機程序在遇到fork()系統調用時，會到達這個fork()系統。


此時，操作系統將創建一個與父進程完全相同的新進程。這意味著前面討論的所有狀態都被複製，包括打開的文件、寄存器狀態和所有內存分配(包括程序代碼)。


系統調用的返回值是進程確定它是現有進程還是新進程的唯一方法。父進程的返回值將是子進程的ID (PID)，而子進程的返回值將為0。


在這一點上，我們說這個進程已經分叉，我們有了如上所述的父子關係。


### 執行

分叉為現有進程提供了啟動新進程的方法，但是如果新進程與父進程不在同一程序中，該怎麼辦呢?這是殻層的情況;當用戶啟動命令時，它需要在一個新的進程中運行，但它與shell無關。


這就是執行系統調用發揮作用的地方。exec將用程序二進制文件中的信息替換當前正在運行的進程的內容。


因此，shell在啟動新程序時所遵循的進程是:首先派生、創建新進程，然後執行(即加載到內存中並執行)它應該運行的程序二進制文件。


### Linux如何處理fork和exec


#### 克隆

在內核中，fork實際上是由一個克隆系統調用實現的。這個克隆接口有效地為Linux內核如何創建進程提供了一個抽象級別。


克隆允許您顯式地指定將新進程的哪些部分複製到新進程中，以及在兩個進程之間共享哪些部分。乍一看這似乎有點奇怪，但它允許我們使用一個非常簡單的接口輕鬆實現線程。


#### 線程

當fork複製我們上面提到的所有屬性時，想像一下，如果除了內存之外，所有的東西都為新進程複製了。這意味著父進程和子進程共享相同的內存，其中包括程序代碼和數據。


> 圖5.4。線程

![](http://www.bottomupcs.com/chapter04/figures/threads.png)

<!--

The memory (including program code and variables) of the process are shared by the threads, but each has its own kernel state, so they can be running different parts of the code at the same time.

----

This hybrid child is called a thread. Threads have a number of advantages over where you might use fork

1. Separate processes can not see each others memory. They can only communicate with each other via other system calls.
  Threads however, share the same memory. So you have the advantage of multiple processes, with the expense of having to use system calls to communicate between them.
  The problem that this raises is that threads can very easily step on each others toes. One thread might increment a variable, and another may decrease it without informing the first thread. These type of problems are called concurrency problems and they are many and varied.
  To help with this, there are userspace libraries that help programmers work with threads properly. The most common one is called POSIX threads or, as it more commonly referred to pthreads
2. Switching processes is quite expensive, and one of the major expenses is keeping track of what memory each process is using. By sharing the memory this overhead is avoided and performance can be significantly increased.

There are many different ways to implement threads. On the one hand, a userspace implementation could implement threads within a process without the kernel having any idea about it. The threads all look like they are running in a single process to the kernel.
This is suboptimal mainly because the kernel is being withheld information about what is running in the system. It is the kernels job to make sure that the system resources are utilised in the best way possible, and if what the kernel thinks is a single process is actually running multiple threads it may make suboptimal decisions.

Thus the other method is that the kernel has full knowledge of the thread. Under Linux, this is established by making all processes able to share resources via the clone system call. Each thread still has associated kernel resources, so the kernel can take it into account when doing resource allocations.

Other operating systems have a hybrid method, where some threads can be specified to run in userspace only ("hidden" from the kernel) and others might be a light weight process, a similar indication to the kernel that the processes is part of a thread group.

####Copy on write
As we mentioned, copying the entire memory of one process to another when fork is called is an expensive operation.

One optimisation is called copy on write. This means that similar to threads above, the memory is actually shared, rather than copied, between the two processes when fork is called. If the processes are only going to be reading the memory, then actually copying the data is unnecessary.

However, when a process writes to its memory, it needs to be a private copy that is not shared. As the name suggests, copy on write optimises this by only doing the actual copy of the memory at the point when it is written to.

Copy on write also has a big advantage for exec. Since exec will simply be overwriting all the memory with the new program, actually copying the memory would waste a lot of time. Copy on write saves us actually doing the copy.

###The init process
We discussed the overall goal of the init process previously, and we are now in a position to understand how it works.

On boot the kernel starts the init process, which then forks and execs the systems boot scripts. These fork and exec more programs, eventually ending up forking a login process.

The other job of the init process is "reaping". When a process calls exit with a return code, the parent usually wants to check this code to see if the child exited correctly or not.

However, this exit code is part of the process which has just called exit. So the process is "dead" (e.g. not running) but still needs to stay around until the return code is collected. A process in this state is called a zombie (the traits of which you can contrast with a mystical zombie!)

A process stays as a zombie until the parent collects the return code with the wait call. However, if the parent exits before collecting this return code, the zombie process is still around, waiting aimlessly to give its status to someone.

In this case, the zombie child will be reparented to the init process which has a special handler that reaps the return value. Thus the process is finally free and the descriptor can be removed from the kernels process table.

####Zombie example

> Example 5.3. Zombie example process
-->

> 進程的內存(包括程序代碼和變數)由線程共享，但每個線程都有自己的內核狀態，因此它們可以同時運行代碼的不同部分。



這個混合子結點被稱為線程。與使用fork相比，線程有很多優勢


1。單獨的進程不能看到彼此的內存。它們只能通過其他系統調用彼此通信。

然而，線程共享相同的內存。因此，您擁有多個進程的優勢，但必須使用系統調用在它們之間進行通信。

這帶來的問題是線程很容易相互踩到對方的腳趾。一個線程可能增加一個變數，另一個線程可能減少它，而不通知第一個線程。這些類型的問題稱為並發問題，它們是多種多樣的。

為瞭解決這個問題，有一些用戶空間庫可以幫助程序員正確地使用線程。最常見的是POSIX線程，或者更常見的是pthreads

2。切換進程非常昂貴，其中一個主要開銷就是跟蹤每個進程使用的內存。通過共享內存，可以避免這種開銷，並顯著提高性能。


實現線程有許多不同的方法。一方面，用戶空間實現可以在進程中實現線程，而內核對此一無所知。這些線程看起來都像是在內核的單個進程中運行。

這是次優的，主要是因為內核被保留了關於系統中正在運行的內容的信息。內核的工作是確保系統資源以儘可能最好的方式得到利用，如果內核認為單個進程實際上在運行多個線程，那麼它可能會做出次優決策。


因此，另一種方法是內核完全瞭解線程。在Linux下，這是通過使所有進程能夠通過克隆系統調用共享資源來實現的。每個線程仍然有相關的內核資源，所以內核在進行資源分配時可以考慮到它。


其他操作系統有一個混合方法，其中一些線程可以指定僅在用戶空間中運行(對內核“隱藏”)，而其他線程可能是一個輕量級進程，這與內核指示進程是線程組的一部分類似。


### 在寫的時候複製

正如我們所提到的，在調用fork時，將一個進程的整個內存複製到另一個進程是一項昂貴的操作。


一種優化方法被稱為寫複製。這意味著，與上面的線程類似，在調用fork時，內存實際上是在兩個進程之間共享的，而不是複製的。如果進程只讀取內存，那麼實際上複製數據是不必要的。


然而，當進程寫入內存時，它需要是一個不共享的私有副本。顧名思義，write on write只在寫入內存時執行內存的實際副本，從而優化了這一點。


抄寫對執行人員也有很大的優勢。由於exec將簡單地用新程序覆蓋所有內存，實際上複製內存將浪費大量時間。在寫的時候複製可以節省我們的實際操作。


### init進程

我們以前討論過init進程的總體目標，現在我們可以理解它是如何工作的。


在啟動時，內核啟動init進程，然後該進程對系統啟動腳本進行分叉和執行。這些分叉和執行更多的程序，最終以分叉登錄進程結束。


init進程的另一項工作是“收穫”。當進程使用返回代碼調用exit時，父進程通常希望檢查這段代碼，看看子進程是否正確退出。


但是，這個退出代碼是剛剛調用exit的進程的一部分。所以這個進程是“死的”(例如，沒有運行)，但是仍然需要停留在那裡直到返回代碼被收集。處於這種狀態的進程稱為殭屍(你可以將其特徵與神秘的殭屍進行對比!)


在父進程通過等待調用收集返回代碼之前，進程一直保持殭屍狀態。但是，如果父進程在收集此返回代碼之前退出，那麼殭屍進程仍然存在，毫無目的地等待將其狀態提供給某人。


在本例中，僵死子進程將被修復到init進程，該進程有一個特殊的處理程序來獲取返回值。因此，進程最終是空閒的，描述符可以從內核進程表中刪除。


#### 殭屍的例子


> 5.3的例子。殭屍示例流程

```
$ cat zombie.c
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
        pid_t pid;

        printf("parent : %d\n", getpid());

        pid = fork();

        if (pid == 0) {
                printf("child : %d\n", getpid());
                sleep(2);
                printf("child exit\n");
                exit(1);
        }

        /* in parent */
        while (1)
        {
                sleep(1);
        }
}

ianw@lime:~$ ps ax | grep [z]ombie
16168 pts/9    S      0:00 ./zombie
16169 pts/9    Z      0:00 [zombie] <defunct>
```

<!--

Above we create a zombie process. The parent process will sleep forever, whilst the child will exit after a few seconds.

Below the code you can see the results of running the program. The parent process (16168) is in state S for sleep (as we expect) and the child is in state Z for zombie. The ps output also tells us that the process is defunct in the process description.[16]

----

[16] The square brackets around the "z" of "zombie" are a little trick to remove the grep processes itself from the ps output. grep interprets everything between the square brackets as a character class, but because the process name will be "grep [z]ombie" (with the brackets) this will not match!

----

##Context Switching

Context switching refers to the process the kernel undertakes to switch from one process to another. XXX ?

##Scheduling
A running system has many processes, maybe even into the hundreds or thousands. The part of the kernel that keeps track of all these processes is called the scheduler because it schedules which process should be run next.

Scheduling algorithms are many and varied. Most users have different goals relating to what they want their computer to do, so this affects scheduling decisions. For example, for a desktop PC you want to make sure that your graphical applications for your desktop are given plenty of time to run, even if system processes take a little longer. This will increase the responsiveness the user feels, as their actions will have more immediate responses. For a server, you might want your web server application to be given priority.

People are always coming up with new algorithms, and you can probably think of your own fairly easily. But there are a number of different components of scheduling.

###Preemptive v co-operative scheduling
Scheduling strategies can broadly fall into two categories

Co-operative scheduling is where the currently running process voluntarily gives up executing to allow another process to run. The obvious disadvantage of this is that the process may decide to never give up execution, probably because of a bug causing some form of infinite loop, and consequently nothing else can ever run.

Preemptive scheduling is where the process is interrupted to stop it to allow another process to run. Each process gets a time-slice to run in; at the point of each context switch a timer will be reset and will deliver and interrupt when the time-slice is over.

We know that the hardware handles the interrupt independently of the running process, and so at this point control will return to the operating system. At this point, the scheduler can decide which process to run next.

This is the type of scheduling used by all modern operating systems.

###Realtime
Some processes need to know exactly how long their time-slice will be, and how long it will be before they get another time-slice to run. Say you have a system running a heart-lung machine; you don't want the next pulse to be delayed because something else decided to run in the system!

Hard realtime systems make guarantees about scheduling decisions like the maximum amount of time a process will be interrupted before it can run again. They are often used in life critical applications like medical, aircraft and military applications.

Soft realtime is a variation on this, where guarantees aren't as strict but general system behaviour is predictable. Linux can be used like this, and it is often used in systems dealing with audio and video. If you are recording an audio stream, you don't want to be interrupted for long periods of time as you will loose audio data which can not be retrieved.

###Nice value
UNIX systems assign each process a nice value. The scheduler looks at the nice value and can give priority to those processes that have a higher "niceness".

A brief look at the Linux Scheduler
The Linux scheduler has and is constantly undergoing many changes as new developers attempt to improve its behaviour.

The current scheduler is known as the O(1) scheduler, which refers to the property that no matter how many processes the scheduler has to choose from, it will choose the next one to run in a constant amount of time[17].

Previous incarnations of the Linux scheduler used the concept of goodness to determine which process to run next. All possible tasks are kept on a run queue, which is simply a linked list of processes which the kernel knows are in a "runnable" state (i.e. not waiting on disk activity or otherwise asleep). The problem arises that to calculate the next process to run, every possible runnable process must have its goodness calculated and the one with the highest goodness ``wins''. You can see that for more tasks, it will take longer and longer to decide which processes will run next.

> Figure 5.5. The O(1) scheduler
-->

上面我們創建了一個殭屍進程。父進程將永遠休眠，而子進程將在幾秒鐘後退出。


在代碼下面，您可以看到運行程序的結果。父進程(16168)為休眠狀態S(正如我們所期望的那樣)，子進程為喪屍狀態Z。ps輸出還告訴我們，流程描述中的流程已經失效。[16]


> [16]“zombie”的“z”周圍的方括弧是一個從ps輸出中刪除grep進程本身的小技巧。grep將方括弧之間的所有內容都解釋為一個字元類，但是由於進程名將是“grep [z]ombie”(帶有方括弧)，因此將不匹配!


## 上下文切換


上下文切換是指內核從一個進程切換到另一個進程的過程。XXX嗎?


## 調度

一個運行中的系統有許多進程，甚至可能有成百上千個。內核中跟蹤所有這些進程的部分稱為調度器，因為它計劃接下來應該運行哪個進程。


調度算法是多種多樣的。大多數用戶對於他們的計算機要做的事情有不同的目標，所以這影響了調度決策。例如，對於桌面PC，您希望確保您的桌面圖形應用程序有足夠的時間運行，即使系統進程需要更長的時間。這將增加用戶的響應能力，因為他們的行動將有更多的即時響應。對於服務器，您可能希望web服務器應用程序獲得優先級。


人們總是會想出新的算法，你可以很容易地想到自己的算法。但是調度有很多不同的組成部分。


## 先發制人的v合作調度

調度策略大致可以分為兩類


合作調度是指當前運行的進程自願放棄執行以允許另一個進程運行。這樣做的明顯缺點是，進程可能決定永遠不放棄執行，這可能是因為錯誤導致了某種形式的無限循環，因此其他任何東西都不能運行。


搶佔式調度是指中斷進程以停止進程以允許另一個進程運行。每個進程都有一個運行的時間片;在每個上下文切換點，計時器將被重置，並在時間片結束時發送和中斷。


我們知道，硬件獨立於運行的進程處理中斷，因此此時控制將返回到操作系統。此時，調度器可以決定接下來運行哪個進程。


這是所有現代操作系統使用的調度類型。


### 實時

有些進程需要確切地知道它們的時間片會有多長，以及在它們得到另一個時間片運行之前會有多長。假設你有一個運行心肺機的系統;您不希望下一個脈衝因為其他東西決定在系統中運行而被延遲!


硬實時系統保證了調度決策，比如進程在重新運行之前被中斷的最大時間。它們經常用於生命關鍵應用，如醫療、飛機和軍事應用。


軟實時是這方面的一種變體，它的保證不那麼嚴格，但一般的系統行為是可以預測的。Linux可以這樣使用，它經常用於處理音頻和視頻的系統。如果您正在錄製音頻流，您不希望被長時間中斷，因為您會丟失無法檢索的音頻數據。


### 好價值

UNIX系統為每個進程分配一個很好的值。調度器查看漂亮的值，並可以優先考慮那些具有較高“良好性”的進程。


簡要介紹一下Linux調度器

隨著新開發人員試圖改進Linux調度器的行為，Linux調度器已經並且正在不斷地進行許多更改。


當前的調度器稱為O(1)調度器，它指的是無論調度器必須從多少進程中選擇，它都會選擇下一個進程，並在一定的時間內運行[17]。


以前版本的Linux調度器使用了好處的概念來決定接下來運行哪個進程。所有可能的任務都保持在運行隊列中，這只是一個進程鏈表，內核知道這些進程處於“可運行”狀態(即不等待磁碟活動或以其他方式休眠)。問題是，要計算下一個要運行的進程，每一個可能運行的進程都必須計算它的良度，而良度最高的進程必須“贏”。您可以看到，對於更多的任務，決定接下來運行哪個進程將花費越來越長的時間。


> 圖5.5。O(1)調度器

![](http://www.bottomupcs.com/chapter04/figures/o1queue.png)

<!--

> A view of how the Linux scheduler manages processes.

In contrast, the O(1) scheduler uses a run queue structure as shown above. The run queue has a number of buckets in priority order and a bitmap that flags which buckets have processes available. Finding the next process to run is a matter of reading the bitmap to find the first bucket with processes, then picking the first process off that bucket's queue. The scheduler keeps two such structures, an active and expired array for processes that are runnable and those which have utilised their entire time slice respectively. These can be swapped by simply modifying pointers when all processes have had some CPU time.

The really interesting part, however, is how it is decided where in the run queue a process should go. Some of the things that need to be taken into account are the nice level, processor affinity (keeping processes tied to the processor they are running on, since moving a process to another CPU in a SMP system can be an expensive operation) and better support for identifying interactive programs (applications such as a GUI which may spend much time sleeping, waiting for user input, but when the user does get around to interacting with it wants a fast response).

> [17] Big-O notation is a way of describing how long an algorithm takes to run given increasing inputs. If the algorithm takes twice as long to run for twice as much input, this is increasing linearly. If another algorithm takes four times as long to run given twice as much input, then it is increasing exponentially. Finally if it takes the same amount of time now matter how much input, then the algorithm runs in constant time. Intuitively you can see that the slower the algorithm grows for more input, the better it is. Computer science text books deal with algorithm analysis in more detail.


##The Shell

On a UNIX system, the shell is the standard interface to handling processes on your system. Once the shell was the primary interface, however modern Linux systems have a GUI and provide a shell via a "terminal application" or similar. The primary job of the shell is to help the user handle starting, stopping and otherwise controlling processes running in the system.

When you type a command at the prompt of the shell, it will fork a copy of itself and exec the command that you have specified.

The shell then, by default, waits for that process to finish running before returning to a prompt to start the whole process over again.

As an enhancement, the shell also allows you to background a job, usually by placing an & after the command name. This is simply a signal that the shell should fork and execute the command, but not wait for the command to complete before showing you the prompt again.

The new process runs in the background, and the shell is ready waiting to start a new process should you desire. You can usually tell the shell to foreground a process, which means we do actually want to wait for it to finish.

XXX : a bit of history about bourne shell
-->
> 介紹Linux調度器如何管理進程。


相反，O(1)調度器使用上面所示的運行隊列結構。運行隊列有一些按優先級順序排列的桶，以及一個標記有可用進程的桶的點陣圖。查找下一個要運行的進程只需讀取點陣圖，找到具有進程的第一個bucket，然後從該bucket的隊列中提取第一個進程。調度程序保留兩個這樣的結構，一個是活動的，一個是過期的，用於可運行的進程，另一個是使用了整個時間片的進程。當所有進程都有一定的CPU時間時，只需修改指針即可進行交換。


然而，真正有趣的部分是如何決定進程應該在運行隊列中位於何處。的一些事情需要考慮好水平,處理器關聯性(保持進程與處理器上運行,從進程轉移到另一個CPU在SMP系統中可以是一個昂貴的操作),更好地支持識別交互式程序(應用程序(如一個GUI可能花很多時間睡覺,等待用戶輸入,但當用戶去與它希望快速響應)交互。


> [17]大o符號是一種描述給定遞增輸入的算法運行時間的方法。如果算法需要兩倍的時間來運行兩倍的輸入，這是線性增加的。如果另一種算法的運行時間是輸入量的兩倍，那麼它的運行時間就是指數級增長。最後，如果現在不管輸入多少，花費的時間都是一樣的，那麼算法就在恆定的時間內運行。直觀上，您可以看到，對於更多的輸入，算法增長越慢，就越好。計算機科學教科書更詳細地處理算法分析。



## 外殼


在UNIX系統中，shell是處理系統上進程的標準接口。曾經shell是主要的接口，但是現代Linux系統有一個GUI，並通過“終端應用程序”或類似的方式提供shell。shell的主要工作是幫助用戶處理系統中運行的進程的啟動、停止和其他控制。


當您在shell提示符處鍵入命令時，它將派生一個自己的副本，並執行您指定的命令。


然後，在默認情況下，shell會等待進程完成運行，然後返回到提示符，重新開始整個進程。


作為一種增強，shell還允許您對作業進行後台處理，通常在命令名後面加上&。這只是一個信號，shell應該派生並執行命令，而不是等待命令完成後再顯示提示符。


新的進程在後台運行，而shell已經準備好等待啟動新的進程。您通常可以告訴shell將進程前台化，這意味著我們實際上希望等待它完成。


關於伯恩·謝爾的一些歷史

<!--

##Signals
Processes running in the system require a way to be told about events that influence them. On UNIX there is infrastructure between the kernel and processes called signals which allows a process to receive notification about events important to it.

When a signal is sent to a process, the kernel invokes a handler which the process must register with the kernel to deal with that signal. A handler is simply a designed function in the code that has been written to specifically deal with interrupt. Often the signal will be sent from inside the kernel itself, however it is also common for one process to send a signal to another process (one form of interprocess communication). The signal handler gets called asynchronously; that is the currently running program is interrupted from what it is doing to process the signal event.

For example, one type of signal is an interrupt (defined in system headers as SIGINT) is delivered to the process when the ctrl-c combination is pressed.

As a process uses the read system call to read input from the keyboard, the kernel will be watching the input stream looking for special characters. Should it see a ctrl-c it will jump into signal handling mode. The kernel will look to see if the process has registered a handler for this interrupt. If it has, then execution will be passed to that function where the function will handle it. Should the process have not registered a handler for this particular signal, then the kernel will take some default action. With ctrl-c the default action is to terminate the process.

A process can choose to ignore some signals, but other signals are not allowed to be ignored. For example, SIGKILL is the signal sent when a process should be terminated. The kernel will see that the process has been sent this signal and terminate the process from running, no questions asked. The process can not ask the kernel to ignore this signal, and the kernel is very careful about which process is allowed to send this signal to another process; you may only send it to processes owned by you unless you are the root user. You may have seen the command kill -9; this comes from the implementation SIGKILL signal. It is commonly known that SIGKILL is actually defined to be 0x9, and so when specified as an argument to the kill program means that the process specified is going to be stopped immediately. Since the process can not choose to ignore or handle this signal, it is seen as an avenue of last resort, since the program will have no chance to clean up or exit cleanly. It is considered better to first send a SIGTERM (for terminate) to the process first, and if it has crashed or otherwise will not exit then resort to the SIGKILL. As a matter of convention, most programs will install a handler for SIGHUP (hangup -- a left over from days of serial terminals and modems) which will reload the program, perhaps to pick up changes in a configuration file or similar.

If you have programmed on a Unix system you would be familiar with segmentation faults when you try to read or write to memory that has not been allocated to you. When the kernel notices that you are touching memory outside your allocation, it will send you the segmentation fault signal. Usually the process will not have a handler installed for this, and so the default action to terminate the program ensues (hence your program "crashes"). In some cases a program may install a handler for segmentation faults, although reasons for doing this are limited.

This raises the question of what happens after the signal is received. Once the signal handler has finished running, control is returned to the process which continues on from where it left off.

###Example
The following simple program introduces a lot of signals to run!

> Example 5.4. Signals Example
-->
## 信號

系統中運行的進程需要知道影響它們的事件。在UNIX上，內核和稱為信號的進程之間有基礎設施，允許進程接收關於對它重要的事件的通知。


當一個信號被發送到一個進程時，內核會調用一個處理器，這個處理器必須註冊到內核來處理這個信號。處理程序只是為處理中斷而編寫的代碼中的一個設計好的函數。通常信號會從內核內部發出，但是一個進程向另一個進程(一種進程間通信形式)發送信號也是很常見的。信號處理器非同步調用;也就是說，當前正在運行的程序與它處理信號事件時所做的工作相中斷。


例如，當按下ctrl-c組合鍵時，一種類型的信號是中斷(在系統標頭中定義為SIGINT)被傳遞到進程。


當進程使用read系統調用從鍵盤讀取輸入時，內核將監視輸入流尋找特殊字元。如果它看到ctrl-c，它將跳到信號處理模式。內核將查看進程是否為這個中斷註冊了處理程序。如果有，那麼執行將被傳遞到函數處理它的地方。如果進程沒有為這個特定的信號註冊一個處理程序，那麼內核將採取一些默認操作。使用ctrl-c，默認操作是終止進程。


進程可以選擇忽略某些信號，但其他信號是不允許忽略的。例如，SIGKILL是應該終止進程時發送的信號。內核將看到進程已經被發送了這個信號，並停止進程運行，沒有任何問題。進程不能要求內核忽略這個信號，內核對於允許哪個進程將這個信號發送到另一個進程非常小心;除非您是根用戶，否則您只能將其發送到您擁有的進程。你可能見過命令kill -9;這來自於SIGKILL信號的實現。眾所周知，SIGKILL實際上被定義為0x9，因此，當作為kill程序的參數指定時，意味著指定的進程將立即停止。由於進程不能選擇忽略或處理這個信號，它被視為最後的手段，因為程序將沒有機會清理或乾淨地退出。最好先向進程發送一個SIGTERM(用於終止)，如果它已經崩潰或無法退出，那麼就求助於SIGKILL。按照慣例，大多數程序都會為SIGHUP(掛起——串口終端和調製解調器時代遺留下來的)安裝一個處理程序，它會重新加載程序，可能是在配置文件或類似文件中獲取更改。


如果您在Unix系統上編程，那麼當您嘗試對未分配給您的內存進行讀寫時，您將熟悉分段錯誤。當內核注意到您正在訪問分配之外的內存時，它將向您發送分割錯誤信號。通常，進程不會為此安裝處理程序，因此終止程序的默認操作會隨之發生(因此程序“崩潰”)。在某些情況下，程序可能會為分割錯誤安裝處理程序，儘管這樣做的原因是有限的。


這就提出了信號接收後會發生什麼問題。一旦信號處理程序完成運行，控制將返回到進程，該進程將從停止的地方繼續運行。


### 例子

下面這個簡單的程序引入了很多信號來運行!


> 5.4的例子。信號的例子

```
$ cat signal.c
#include <stdio.h>
#include <unistd.h>
#include <signal.h>

void sigint_handler(int signum)
{
        printf("got SIGINT\n");
}

int main(void)
{
        signal(SIGINT, sigint_handler);
        printf("pid is %d\n", getpid());
        while (1)
                sleep(1);
}
$ gcc -Wall -o signal signal.c
$ ./signal
pid is 2859
got SIGINT #press ctrl-c 
           #press ctrl-z
[1]+  Stopped                 ./signal

$ kill -SIGINT 2859
$ fg
./signal
got SIGINT
Quit #press ctrl-\

$
```

<!--

We have simple program that simply defines a handler for the SIGINT signal, which is sent when the user presses ctrl-c. All the signals for the system are defined in signal.h, including the signal function which allows us to register the handling function.

The program simply sits in a tight loop doing nothing until it quits. When we start the program, we try pressing ctrl-c to make it quit. Rather than taking the default action, or handler is invoked and we get the output as expected.

We then press ctrl-z which sends a SIGSTOP which by default puts the process to sleep. This means it is not put in the queue for the scheduler to run and is thus dormant in the system.

As an illustration, we use the kill program to send the same signal from another terminal window. This is actually implemented with the kill system call, which takes a signal and PID to send to (this function is a little misnamed because not all signals do actually kill the process, as we are seeing, but the signal function was already taken to register the handler). As the process is stopped, the signal gets queued for the process. This means the kernel takes note of the signal and will deliver it when appropriate.

At this point we wake the process up by using the command fg. This actually sends a SIGCONT signal to the process, which by default will wake the process back up. The kernel knows to put the process on the run queue and give it CPU time again. We see at this point the queued signal is delivered.

In desperation to get rid of the program, we finally try ctrl-\ which sends a SIGQUIT (abort) to the process. But if the process has aborted, where did the Quit output come from?

You guessed it, more signals! When a parent child has a process that dies, it gets a SIGCHLD signal back. In this case the shell was the parent process and so it got the signal. Remember how we have the zombie process that needs to be reaped with the wait call to get the return code from the child process? Well another thing it also gives the parent is the signal number that the child may have died from. Thus the shell knows that child process died from a SIGABRT and as an informational service prints as much for the user (the same process happens to print out "Segmentation Fault" when the child process dies from a SIGSEGV).

You can see how in even a simple program, around 5 different signals were used to communicate between processes and the kernel and keep things running. There are many other signals, but these are certainly amongst the most common. Most have system functions defined by the kernel, but there are a few signals reserved for users to use for their own purposes within their programs (SIGUSR).
-->
我們有一個簡單的程序，它簡單地為SIGINT信號定義一個處理程序，當用戶按ctrl-c時發送該信號。系統的所有信號都是在信號中定義的。h，包括允許我們註冊處理函數的信號函數。


該程序只是處於一個緊密的循環中，在退出之前什麼也不做。當我們啟動程序時，我們嘗試按ctrl-c使它退出。不是採取默認的操作，而是調用處理程序，並按預期得到輸出。


然後按ctrl-z，它會發送一個SIGSTOP，該SIGSTOP在默認情況下會讓進程進入休眠狀態。這意味著它不會被放到調度程序運行的隊列中，因此在系統中處於休眠狀態。


作為一個例子，我們使用kill程序從另一個終端窗口發送相同的信號。這實際上是通過kill系統調用實現的，該調用接受一個信號和PID發送到(這個函數的名稱有點錯誤，因為並不是所有的信號都真正殺死進程，正如我們所看到的，但是信號函數已經被用來註冊處理器)。當進程停止時，信號將排隊等待進程。這意味著內核會注意到信號，並在適當的時候發送它。


此時，我們使用命令fg喚醒流程。這實際上向進程發送了SIGCONT信號，默認情況下，這將喚醒進程。內核知道將進程放到運行隊列中，並給它一次CPU時間。我們在這裡看到排隊的信號被發送。


為了擺脫這個程序，我們最後嘗試ctrl-\，它向進程發送SIGQUIT (abort)。但如果進程已經中止，退出輸出從何而來?


你猜對了，更多的信號!當父進程終止時，它會返回一個SIGCHLD信號。在本例中，shell是父進程，因此它得到了信號。還記得我們是如何利用等待調用來獲取子進程返回代碼的嗎?另一件事它也給了父母一個信號，孩子可能死於這個信號。因此，shell知道子進程死於SIGABRT，而作為信息性服務，它會為用戶打印同樣多的內容(當子進程死於SIGSEGV時，同樣的進程會打印出“分割錯誤”)。


您可以看到，即使在一個簡單的程序中，也有大約5種不同的信號被用來在進程和內核之間通信，並保持運行。還有很多其他的信號，但這些肯定是最常見的。大多數都有由內核定義的系統功能，但是有一些信號保留給用戶在他們的程序(SIGUSR)中用於他們自己的目的。