# 第 1 章 ： 通用 Unix 和高級 C 語言 (General Unix and Advanced C)

## 所有東西都是文件 !  (Everything is a file ! )

類unix系統 (如Linux或BSD) 經常引用的一個原則是，一切都是文件。

<!-- 
An often-quoted tenet of UNIX-like systems such as Linux or BSD is everything is a file.

-->

想像一個文件在一個熟悉的環境中，比如一個文字處理器。對於這個虛構的字處理文件，我們可以使用兩種基本操作:

1. 讀取它(從文字處理器中保存的現有數據)。
2. 寫入(來自用戶的新數據)。

考慮一些附加到計算機上的常見事物，以及它們如何與我們的基本文件操作相關聯:

1. 屏幕
2. 鍵盤
3. 一台打印機
4. 一個光盤

<!-- 
Imagine a file in the context of something familiar like a word processor. There are two fundamental operations we could use on this imaginary word processing file:

1. Read it (existing saved data from the word processor).
2. Write to it (new data from the user).

Consider some of the common things attached to a computer and how they relate to our fundamental file operations:

1. The screen
2. The keyboard
3. A printer
4. A CD-ROM

-->


屏幕和打印機都類似於只寫的文件，但是信息不是存儲在磁碟上，而是顯示在屏幕上的點或頁面上的行。鍵盤就像一個只讀文件，數據來自用戶提供的擊鍵。CD-ROM與此類似，但數據不是隨機從用戶那裡獲得的，而是直接存儲在磁碟上。

<!--
The screen and printer are both like a write-only file, but instead of being stored as bits on a disk the information is displayed as dots on a screen or lines on a page. The keyboard is like a read only file, with the data coming from keystrokes provided by the user. The CD-ROM is similar, but rather than randomly coming from the user the data is stored directly on the disk.

-->

因此，文件的概念是對數據接收器或數據源的良好抽象。因此，它是對所有可能連接到計算機上的設備的完美抽象。這種認識是UNIX的強大功能，在整個平台的設計中都很明顯。向程序員提供硬件抽象是操作系統的基本角色之一。


說抽象是支撐所有現代計算的主要概念，這可能並不過分。沒有人能夠理解從設計現代用戶界面到現代CPU的內部工作原理的一切，更不用說自己構建它了。對於程序員來說，抽象是允許我們合作和發明的通用語言。


學習在抽象中導航可以讓你更深入地瞭解如何以最好和最創新的方式使用抽象。在這本書中，我們關注的是底層的抽象;應用程序和操作系統之間，操作系統和硬件之間。在這之上還有更多的層次，每一層都配得上自己的書。隨著這些章節的進展，您將有望深入瞭解現代操作系統所呈現的抽象。

<!-- 
Thus the concept of a file is a good abstraction of either a sink for, or source of, data. As such it is an excellent abstraction of all the devices one might attach to the computer. This realisation is the great power of UNIX and is evident across the design of the entire platform. It is one of the fundamental roles of the operating system to provide this abstraction of the hardware to the programmer.

It is probably not too much of a stretch to say abstraction is the primary concept that underpins all modern computing. No one person can understand everything from designing a modern user-interface to the internal workings of a modern CPU, much less build it all themselves. To programmers, abstractions are the common language that allows us to collaborate and invent.

Learning to navigate across abstractions gives one greater insight into how to use the abstractions in the best and most innovative ways. In this book, we are concerned with abstractions at the lowest layers; between applications and the operating system and the operating system and hardware. Many more layers lie above this, each worthy of their own books. As these chapters progress, you will hopefully gain some insight into the abstractions presented by a modern operating system.

-->

> 圖1.1 - 抽象 (Abstraction)

![Figure 1.1. Abstraction](http://www.bottomupcs.com/chapter00/figures/abstraction.png)

## 實現抽象 (Implementing abstraction)

通常，抽象是由通常稱為應用程序編程接口(API)的東西實現的。API是一個有點模糊的術語，在各種編程工作的上下文中意味著不同的東西。從根本上說，程序員設計一組函數 API 並記錄它們的接口和功能，但其實際實現對使用者則是不透明的 (使用者不須知道實現方式)。

例如，許多大型web應用程序提供了通過HTTP訪問的API。通過這種方法訪問數據肯定會觸發許多複雜的遠程過程調用、數據庫查詢和數據傳輸，但這些對只接收契約數據的最終用戶來說是不透明的。

熟悉物件導向語言 (如Java、Python或c++) 的人應該熟悉類別提供的抽象。方法為《類別》提供接口，但其實現則被抽象化而隱藏了。

<!-- 
In general, abstraction is implemented by what is generically termed an Application Programming Interface (API). API is a somewhat nebulous term that means different things in the context of various programming endeavours. Fundamentally, a programmer designs a set of functions and documents their interface and functionality with the principle that the actual implementation providing the API is opaque.

For example, many large web applications provide an API accessible via HTTP. Accessing data via this method surely triggers many complicated series of remote procedure calls, database queries and data transfers, all of which are opaque to the end user who simply receives the contracted data.

Those familiar with object-oriented languages such as Java, Python or C++ would be familiar with the abstraction provided by classes. Methods provide the interface to the class, but abstract the implementation.

-->

## 用C實現抽象 (Implementing abstraction with C)

在 Linux 內核和其他大型 C 代碼庫中，缺少物件導向的內置概念的一種常用方法是函數指針。學習閲讀這個習慣用法是瀏覽大多數大型 C 代碼庫的關鍵。通過理解如何閲讀代碼中提供的抽象，可以理解內部 API 設計。

<!-- 
A common method used in the Linux kernel and other large C code bases, which lack a built-in concept of object-orientation, is function pointers. Learning to read this idiom is key to navigating most large C code bases. By understanding how to read the abstractions provided within the code an understanding of internal API designs can be built.

-->

> 例 1.1。帶有函數指針的抽象物件 (Abstraction with function pointers)

```c
#include <stdio.h>

/* The API to implement */
struct greet_api
{
	int (*say_hello)(char *name);
	int (*say_goodbye)(void);
};

/* Our implementation of the hello function */
int say_hello_fn(char *name)
{
	printf("Hello %s\n", name);
	return 0;
}

/* Our implementation of the goodbye function */
int say_goodbye_fn(void)
{
	printf("Goodbye\n");
	return 0;
}

/* A struct implementing the API */
struct greet_api greet_api =
{
	.say_hello = say_hello_fn,
	.say_goodbye = say_goodbye_fn
};

/* main() doesn't need to know anything about how the
 * say_hello/goodbye works, it just knows that it does */
int main(int argc, char *argv[])
{
	greet_api.say_hello(argv[1]);
	greet_api.say_goodbye();

	printf("%p, %p, %p\n", greet_api.say_hello, say_hello_fn, &say_hello_fn);

	exit(0);
}

```

上面的代碼是在 Linux 內核和其他 C 程序中反覆使用的最簡單的構造示例。讓我們來看看一些具體的元素。


我們從定義API (struct greet_api)的結構開始。名稱用指針標記括在括弧中的函數描述了一個函數指針 [1]。函數指針描述它必須指向的函數的原型;將其指向沒有正確返回類型或參數的函數，會產生成編譯器警告;如果留在代碼中，可能會導致錯誤的操作或崩潰。


然後是 API 的實現。通常，對於更複雜的功能，您會看到一個習語，其中API實現函數將僅作為其他函數的包裝器，這些函數通常由一個或兩個下劃線構成 (即 say_hello_fn() 將調用另一個函數_say_hello_function())。這有幾個用途;一般來說，它涉及到將更簡單和更小的部分 (例如，編組或檢查參數) 與更複雜的實現分離，這通常會簡化內部工作中發生重大變化的路徑，同時確保 API 保持不變。然而，我們的實現非常簡單，甚至不需要它自己去實現功能。在各種項目中，單下劃線、雙下劃線甚至三下劃線的函數首碼意味著不同的含義，但通常它是一個直觀的警告，即函數不應該直接 “越過” API 去調用。

其次，我們在 struct greet_api greet_api 中填寫函數指針。函數的名稱是指針;因此，不需要獲取函數的地址(即 &say_hello_fn)。


最後，我們可以通過 main 結構調用 API 函數。


在瀏覽原代碼時，您會經常看到這個習慣用法。下面來自include/linux/virtio.h 在 Linux 內核源碼的小例子可說明這種習慣用法:

<!-- 
Code such as the above is the simplest example of constructs used repeatedly throughout the Linux Kernel and other C programs. Let's have a look at some specific elements.

We start out with a structure that defines the API (struct greet_api). The functions whose names are encased in parentheses with a pointer marker describe a function pointer[1]. The function pointer describes the prototype of the function it must point to; pointing it at a function without the correct return type or parameters will generate a compiler warning at least; if left in code will likely lead to incorrect operation or crashes.

We then have our implementation of the API. Often for more complex functionality you will see an idiom where API implementation functions will only be a wrapper around other functions that are conventionally prepended with one or or two underscores[2] (i.e. say_hello_fn() would call another function _say_hello_function()). This has several uses; generally it relates to having simpler and smaller parts of the API (marshalling or checking arguments, for example) separate from more complex implementation, which often eases the path to significant changes in the internal workings whilst ensuring the API remains constant. Our implementation is very simple, however, and doesn't even need it's own support functions. In various projects, single-, double- or even triple-underscore function prefixes will mean different things, but universally it is a visual warning that the function is not supposed to be called directly from "beyond" the API.

Second to last, we fill out the function pointers in struct greet_api greet_api. The name of the function is a pointer; therefore there is no need to take the address of the function (i.e. &say_hello_fn).

Finally we can call the API functions through the structure in main.

You will see this idiom constantly when navigating the source code. The tiny example below is taken from include/linux/virtio.h in the Linux kernel source to illustrate:

-->

> 例 1.2。在 include/linux/virtio.h 的抽象定義 (Abstraction in include/linux/virtio.h)

```c
/**
 * virtio_driver - operations for a virtio I/O driver
 * @driver: underlying device driver (populate name and owner).
 * @id_table: the ids serviced by this driver.
 * @feature_table: an array of feature numbers supported by this driver.
 * @feature_table_size: number of entries in the feature table array.
 * @probe: the function to call when a device is found.  Returns 0 or -errno.
 * @remove: the function to call when a device is removed.
 * @config_changed: optional function to call when the device configuration
 *    changes; may be called in interrupt context.
 */
struct virtio_driver {
        struct device_driver driver;
        const struct virtio_device_id *id_table;
        const unsigned int *feature_table;
        unsigned int feature_table_size;
        int (*probe)(struct virtio_device *dev);
        void (*scan)(struct virtio_device *dev);
        void (*remove)(struct virtio_device *dev);
        void (*config_changed)(struct virtio_device *dev);
#ifdef CONFIG_PM
        int (*freeze)(struct virtio_device *dev);
        int (*restore)(struct virtio_device *dev);
#endif
};
```

只需要模糊地理解這個結構就是虛擬 I/O 設備的描述。我們可以看到，這個 API 的用戶(設備驅動程序的作者)期望提供一些功能，這些功能將在系統運行期間 (探測新硬件時，當硬件被刪除時，等等) 在各種條件下被調用。它還包含一系列數據;結構中應填入相關數據。

從這樣的描述符開始通常是理解各種內核代碼層的最簡單方法。

<!-- 
It's only necessary to vaguely understand that this structure is a description of a virtual I/O device. We can see the user of this API (the device driver author) is expected to provide a number of functions that will be called under various conditions during system operation (when probing for new hardware, when hardware is removed, etc.). It also contains a range of data; structures which should be filled with relevant data.

Starting with descriptors like this is usually the easiest way to begin understanding the various layers of kernel code.

-->

## 庫 (Libraries)

庫有兩個角色來說明抽象。

* 允許程序員重用經常訪問的代碼。
* 充當一個黑盒子，為程序員實現功能。

例如,函式庫在 JPEG 文件中實現訪問原始數據,許多程序要訪問圖象文件都可以使用相同的庫，程序員構建這些程序不需要擔心 JPEG 文件格式的具體細節,但可以集中焦點在他們想要做什麼圖像功能上。

UNIX平台的標準庫一般稱為 libc。它提供了系統的基本接口:基本調用，如read()、write()和printf()。這個 API 由一個叫做 POSIX 的規範完整地描述。它可以在線免費獲得，並描述了構成標準 UNIX API 的許多調用。


大多數 UNIX 平台廣泛地遵循 POSIX 標準，儘管它們之間的差別很小，但有時很重要 (因此各種 GNU autotools 的複雜性，它們常常試圖為您抽象這些差異)。Linux 有許多 POSIX 沒有指定的接口;編寫專門使用它們的應用程序會降低應用程序的可移植性。


庫是具有許多細節的基本抽象。後面的章節將更詳細地描述庫是如何工作的。

<!-- 
Libraries have two roles which illustrate abstraction.

* Allow programmers to reuse commonly accessed code.
* Act as a black box implementing functionality for the programmer.

For example, a library implementing access to the raw data in JPEG files has both the advantage that the many programs that wish to access image files can all use the same library and the programmers building these programs do not need to worry about the exact details of the JPEG file format, but can concentrate their efforts on what their program wants to do with the image.

The standard library of a UNIX platform is generically referred to as libc. It provides the basic interface to the system: fundamental calls such as read(), write() and printf(). This API is described in its entirety by a specification called POSIX. It is freely available online and describes the many calls that make up the standard UNIX API.

Most UNIX platforms broadly follow the POSIX standard, though often differ in small but sometimes important ways (hence the complexity of the various GNU autotools, which often try to abstract away these differences for you). Linux has many interfaces that are not specified by POSIX; writing applications that use them exclusively will make your application less portable.

Libraries are a fundamental abstraction with many details. Later chapters will describe how libraries work in much greater detail.

-->

## 文件描述符 (File Descriptors)

UNIX程序員首先要瞭解的一件事是，每個運行的程序都從三個已經打開的文件開始:

<!-- 
One of the first things a UNIX programmer learns is that every running program starts with three files already opened:

-->

> 表 1.1 - Unix提供的標準文件 (Standard Files Provided by Unix)

描述性名稱 |	短名稱 | 文件號碼	| 描述 
----------|-------|---------------|-------------------
標準輸入	 |	stdin	 |	0	 |	從鍵盤輸入
標準輸出 	 |	stdout	 |	1	 |	輸出到控制台
標準錯誤	 |	stderr	 |	2	 | 錯誤輸出到控制台

> 圖1.2 - 默認的 Unix 文件 (Default Unix Files)

![Figure 1.2. Default Unix Files](http://www.bottomupcs.com/chapter00/figures/stdfds.png)

這就提出了一個問題:一個打開的文件代表什麼?打開調用返回的值稱為文件描述符，本質上是內核保存的打開文件數組的索引。

<!-- 
This raises the question of what an open file represents. The value returned by an open call is termed a file descriptor and is essentially an index into an array of open files kept by the kernel.

-->

> 圖 1.3 - 抽象 (Abstraction)

![Figure 1.3. Abstraction](http://www.bottomupcs.com/chapter00/figures/file-descriptors.png)

> 文件描述符是內核存儲的文件描述符表的索引。內核創建一個文件描述符以響應一個打開的調用，並將文件描述符與底層類文件對象的抽象相關聯，這些對象是實際的硬件設備、文件系統或其他東西。因此，進程的讀寫調用引用文件描述符被內核路由到正確的位置，從而最終完成一些有用的工作。

簡而言之，文件描述符是進入內核底層硬件抽象的閘道。物理設備抽象的總體視圖如圖1.3“抽象”所示。


從最低級別開始，操作系統要求程序員創建一個設備驅動程序，以便能夠與硬件設備通信。這個設備驅動程序被編寫到內核提供的API中，就像在示例1.2中一樣，“包含/linux/virtio.h中的抽象”;設備驅動程序將提供內核根據各種需求調用的一系列函數。在上面的簡化示例中，我們可以看到驅動程序提供了一個讀寫函數，該函數將被調用，以響應文件描述符上的類似操作。設備驅動程序知道如何將這些通用請求轉換為特定設備的特定請求或命令。


為了向用戶空間提供抽象，內核通過設備層提供了一個文件接口。主機上的物理設備由特殊文件系統(如/dev)中的文件表示。在類 UNIX 系統中，所謂的設備節點具有所謂的主編號和次編號，這允許內核將特定的節點與其底層驅動程序關聯起來。這些可以通過ls識別，如示例1.3 “主和次編號示例” 所示。

<!-- 
> File descriptors are an index into a file descriptor table stored by the kernel. The kernel creates a file descriptor in response to an open call and associates the file descriptor with some abstraction of an underlying file-like object, be that an actual hardware device, or a file system or something else entirely. Consequently a process's read or write calls that reference that file descriptor are routed to the correct place by the kernel to ultimately do something useful.

In short, the file descriptor is the gateway into the kernel's abstractions of underlying hardware. An overall view of the abstraction for physical-devices is shown in Figure 1.3, “Abstraction”.

Starting at the lowest level, the operating system requires a programmer to create a device driver to be able to communicate with a hardware device. This device driver is written to an API provided by the kernel just like in Example 1.2, “Abstraction in include/linux/virtio.h”; the device driver will provide a range of functions which are called by the kernel in response to various requirements. In the simplified example above, we can see the drivers provide a read and write function that will be called in response to the analogous operations on the file descriptor. The device driver knows how to convert these generic requests into specific requests or commands for a particular device.

To provide the abstraction to user-space, the kernel provides a file-interface via what is generically termed a device layer. Physical devices on the host are represented by a file in a special file system such as /dev. In UNIX-like systems, so-called device-nodes have what are termed a major and a minor number, which allow the kernel to associate particular nodes with their underlying driver. These can be identified via ls as illustrated in Example 1.3, “Example of major and minor numbers”.

-->

> 例1.3。主編號和次編號的例子 (Example of major and minor numbers)

```bash
$ ls -l /dev/null /dev/zero /dev/tty
crw-rw-rw- 1 root root 1, 3 Aug 26 13:12 /dev/null
crw-rw-rw- 1 root root 5, 0 Sep  2 15:06 /dev/tty
crw-rw-rw- 1 root root 1, 5 Aug 26 13:12 /dev/zero
```

這將我們帶到文件描述符，它是用戶空間用來與底層設備通信的句柄。從廣義上講，當文件被打開時，內核使用路徑信息將文件描述符映射到提供適當讀寫 API 等內容的對象。當此打開對應設備時 (上面的/dev/sr0)，打開的設備節點的主節點和次節點提供了內核需要的信息，以找到正確的設備驅動程序並完成映射。然後，內核將知道如何將進一步調用路由到設備驅動程序提供的底層函數，比如 read。

非設備文件的操作方式類似，不過中間有更多層。這裡的抽象是掛載點;掛載文件系統的雙重目的是設置映射，以便文件系統知道提供存儲的底層設備，而內核知道在掛載點下打開的文件應該定向到文件系統驅動程序。與設備驅動程序一樣，文件系統被寫入內核提供的特定通用文件系統API。

在現實生活中，確實有許多其他層次使畫面複雜化。例如，內核將竭盡全力從磁碟中緩存儘可能多的數據，以便在其他情況下不受影響的內存中進行緩存;這提供了許多速度優勢。它還將嘗試以最有效的方式組織設備訪問;例如，嘗試命令磁碟訪問，以確保物理上存儲在一起的數據被檢索到一起，即使請求不是按順序到達。此外，許多設備屬於更通用的類，比如USB或SCSI設備，它們提供自己的抽象層供寫入。因此，文件系統不會直接寫到設備上，而是要經過這麼多層。理解內核就是理解這些api是如何相互聯繫和共存的。

<!-- 
This brings us to the file descriptor, which is the handle user-space uses to talk to the underlying device. In a broad sense, what happens when a file is opened is that the kernel is using the path information to map the file descriptor with something that provides an appropriate read and write, etc., API. When this open is for a device (/dev/sr0 above), the major and minor number of the opened device node provides the information the kernel needs to find the correct device driver and complete the mapping. The kernel will then know how to route further calls such as read to the underlying functions provided by the device driver.

A non-device file operates similarly, although there are more layers in between. The abstraction here is the mount point; mounting a file system has the dual purpose of setting up a mapping so the file system knows the underlying device that provides the storage and the kernel knows that files opened under that mount-point should be directed to the file system driver. Like device drivers, file systems are written to a particular generic file system API provided by the kernel.

There are indeed many other layers that complicate the picture in real-life. For example, the kernel will go to great efforts to cache as much data from disks as possible in otherwise-free memory; this provides many speed advantages. It will also try to organise device access in the most efficient ways possible; for example trying to order disk-access to ensure data stored physically close together is retrieved together, even if the requests did not arrive in sequential order. Further, many devices are of a more generic class such as USB or SCSI devices which provide their own abstraction layers to write to. Thus, rather than writing directly to devices, file systems will go through these many layers. Understanding the kernel is to understand how these many APIs interrelate and coexist.

-->

## 外殼 (The Shell)

shell是與操作系統交互的門戶。bash、zsh,csh或任何其他的貝殼,它們都從根本上只有一個主要任務——允許您執行程序(你將開始瞭解殻牌實際上這當我們談到的一些內部操作系統)。


但是shell不僅僅允許您簡單地執行一個程序。它們具有重定向文件、允許您同時執行多個程序和腳本完成程序的強大功能。這些都回到了“一切都是文件”的習慣用法。

<!-- 
The shell is the gateway to interacting with the operating system. Be it bash, zsh, csh or any of the many other shells, they all fundamentally have only one major task — to allow you to execute programs (you will begin to understand how the shell actually does this when we talk about some of the internals of the operating system later).

But shells do much more than allow you to simply execute a program. They have powerful abilities to redirect files, allow you to execute multiple programs simultaneously and script complete programs. These all come back to the everything is a file idiom.

-->

## 重定向 (Redirection)

通常，我們不希望在“文件描述符”一節中提到的標準文件描述符指向它們的默認位置。例如，您可能希望將程序的所有輸出捕獲到磁碟上的文件中，或者，讓它從先前準備的文件中讀取命令。另一個有用的任務可能是將一個程序的輸出傳遞給另一個程序的輸入。有了操作系統，shell使這一切變得更加容易。

<!-- 
Often we do not want the standard file descriptors mentioned in the section called “File Descriptors” to point to their default places. For example, you may wish to capture all the output of a program into a file on disk or, alternatively, have it read its commands from a file you prepared earlier. Another useful task might like to pass the output of one program to the input of another. With the operating system, the shell facilitates all this and more.

-->

> 表1.2。標準外殼重定向設備 (Standard Shell Redirection Facilities)

名稱 | 命令	| 描述	| 示例
-----|-----|-------|------------------------------
重定向文件 	| 	> filename	| 從標準輸出所有輸出並將其放入文件名。注意，使用>>將附加到文件，而不是覆蓋它		| 	ls > filename
從文件讀取	| 	< filename	| 將文件中的所有數據複製到程序	| echo < filename
管道 pipe |  `program1 ｜ program2` | 	從程序1的標準中取出所有東西，並將其傳遞給程序2	| `ls ｜ more`

## 實作管道 Implementing pipe

ls | more的實現只是抽象能力的另一個例子。這裡發生的基本情況是，與其將標準輸出的文件描述符與某種底層設備(例如控制台，用於輸出到終端)關聯起來，還不如將描述符指向內核提供的內存緩衝區(通常稱為管道)。這裡的技巧是，另一個進程可以將其標準輸入與相同緩衝區的另一端相關聯，並有效地消耗另一個進程的輸出。這在圖1.4“運行中的管道”中進行了說明。

<!-- 
The implementation of ls | more is just another example of the power of abstraction. What fundamentally happens here is that instead of associating the file descriptor for the standard-output with some sort of underlying device (such as the console, for output to the terminal), the descriptor is pointed to an in-memory buffer provided by the kernel commonly termed a pipe. The trick here is that another process can associate its standard input with the other side of this same buffer and effectively consume the output of the other process. This is illustrated in Figure 1.4, “A pipe in action”.
-->

> 圖1.4。正在使用的管道 (A pipe in action)

![](http://www.bottomupcs.com/chapter00/figures/pipe.png)

> 管道是連接兩個進程的內存緩衝區。文件描述符指向管道對象，管道對象緩衝(通過寫入)發送給它的數據(通過讀取)。

<!-- 
> The pipe is an in-memory buffer that connects two processes together. file descriptors point to the pipe object, which buffers data sent to it (via a write) to be drained (via a read).

-->


內核存儲對管道的寫操作，直到另一端的相應讀操作耗盡緩衝區為止。這是一個非常強大的概念，是類unix操作系統中進程間通信或IPC的基本形式之一。管道不僅允許數據傳輸;它可以作為一個信號通道。如果進程讀取空管道，它將默認阻塞或進入休眠狀態，直到有一些可用的數據(這在第5章“進程”中有更深入的討論)。因此，兩個進程可以使用管道來通信，通過寫入一個位元組的數據就可以執行某些操作;與實際數據重要不同的是，管道中任何數據的存在都可能發出一條消息。例如，一個進程請求另一個進程打印一個文件——這需要一些時間。兩個進程可以在它們之間建立管道，請求進程在空管道上進行讀取;由於為空，調用阻塞，進程不會繼續。打印完成後，另一個進程可以向管道中寫入消息，從而有效地喚醒請求進程並發出完成工作的信號。

允許進程像這樣在彼此之間傳遞數據，產生了另一種常見的UNIX小工具的習慣。連結這些小工具以完成某項特定任務，可以提供單個整體工具無法提供的靈活性。

<!-- 
Writes to the pipe are stored by the kernel until a corresponding read from the other side drains the buffer. This is a very powerful concept and is one of the fundamental forms of inter-process communication or IPC in UNIX-like operating systems. The pipe allows more than just a data transfer; it can act as a signaling channel. If a process reads an empty pipe, it will by default block or be put into hibernation until there is some data available (this is discussed in much greater depth in Chapter 5, The Process). Thus two processes may use a pipe to communicate that some action has been taken just by writing a byte of data; rather than the actual data being important, the mere presence of any data in the pipe can signal a message. Say for example one process requests that another print a file — something that will take some time. The two processes may set up a pipe between themselves where the requesting process does a read on the empty pipe; being empty, that call blocks and the process does not continue. Once the print is done, the other process can write a message into the pipe, which effectively wakes up the requesting process and signals the work is done.

Allowing processes to pass data between each other like this springs another common UNIX idiom of small tools doing one particular thing. Chaining these small tools gives flexibility that a single monolithic tool often can not.

-->
