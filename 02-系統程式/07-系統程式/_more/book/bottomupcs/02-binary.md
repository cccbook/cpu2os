
# 第 2 章 : 二進制和數字表示 (Binary and Number Representation)

## 二進制 : 計算的基礎 (Binary — the basis of computing)

### 二進制理論 (Binary Theory)

二進制是一個基數為2的數字系統，它使用兩個相互排斥的狀態來表示信息。二進制數是由稱為位的元素組成的，每個位都可以處於兩種可能的狀態之一。通常，我們用數字1和0來表示它們。我們也討論了它們的真假。電性上，這兩種狀態可以用高電壓和低電壓或某種形式的開關來表示。


我們構建二進制數的方式與我們在傳統的基數10系統中構建數字的方式相同。然而，與1的列、10的列、100的列(等等)不同的是，我們有1的列、2的列、4的列、8的列等等，如下圖所示。

<!--
Binary is a base-2 number system that uses two mutually exclusive states to represent information. A binary number is made up of elements called bits where each bit can be in one of the two possible states. Generally, we represent them with the numerals 1 and 0. We also talk about them being true and false. Electrically, the two states might be represented by high and low voltages or some form of switch turned on or off.

We build binary numbers the same way we build numbers in our traditional base 10 system. However, instead of a one's column, a 10's column, a 100's column (and so on) we have a one's column, a two's columns, a four's column, an eight's column, and so on, as illustrated below.

-->

> 表2.1。二進制 Binary

2... | 2<sup>6</sup> | 2<sup>5</sup> | 2<sup>4</sup> | 2<sup>3</sup> | 2<sup>2</sup> | 2<sup>1</sup> | 2<sup>0</sup>
-----|----|----|----|----|----|----|----
...  | 64 | 32 | 16 | 8  | 4  | 2  | 1

例如，為了表示以10為底的數字203，我們知道我們把3放在1的列上，0放在10的列上，2放在100的列上。這是用下面表中的指數表示的。

<!--
For example, to represent the number 203 in base 10, we know we place a 3 in the 1's column, a 0 in the 10's column and a 2 in the 100's column. This is expressed with exponents in the table below.

-->

> 表2.2。203以10為底 (203 in base 10)

10<sup>2</sup> |10<sup>1</sup> | 10<sup>0</sup>
----|----|----
2   | 0  | 3

或者,換句話說, 2 × 10<sup>2</sup> + 3 × 10<sup>0</sup> = 200 + 3 = 203. 為了在二進制中表示相同的東西，我們會有下面的表格。

<!--
Or, in other words, 2 × 10<sup>2</sup> + 3 × 10<sup>0</sup> = 200 + 3 = 203. To represent the same thing in binary, we would have the following table.

> 表 2.3。 203以2為底 (203 in base 2)

2<sup>7<sup> |2<sup>6<sup> |2<sup>5<sup> |2<sup>4<sup> |2<sup>3<sup> |2<sup>2<sup> |2<sup>1<sup> |2<sup>0<sup>
--|--|--|--|--|--|--|--
1 |1 |0 |0 |1 |0 |1 |1

這等於 2<sup>7</sup> + 2<sup>6</sup> + 2<sup>3</sup>+2<sup>1</sup> + 2<sup>0</sup> = 128 + 64 + 8 + 2 + 1 = 203.

<!--
That equates to 2<sup>7</sup> + 2<sup>6</sup> + 2<sup>3</sup>+2<sup>1</sup> + 2<sup>0</sup> = 128 + 64 + 8 + 2 + 1 = 203.

-->

### 計算的基礎 The basis of computing

你可能想知道，一個簡單的數字是計算機所能做的所有神奇事情的基礎。信不信由你，確實如此!處理器在您的計算機有一個複雜但最終有限的指令集,它可以執行等價值觀,乘法,等。從本質上講,這些指令被分配一個號碼,這樣整個程序(添加這個,乘以,除以這個等等)可以表示為一個只是一串數字。例如，如果處理器知道操作2是加法操作，那麼252可能意味著“添加5和2並將輸出存儲在某處”。當然，實際情況要複雜得多(參見第3章，計算機體系結構)，但簡言之，這就是計算機。


在打孔卡的時代，人們可以通過觀察卡上的孔來觀察組成程序流的0和1。當然，這個過程通過小的磁性粒子的極性很快地(磁帶、磁碟)被儲存起來，直到今天，我們的口袋裏還能裝下難以想像的大量數據。


把這些數字轉換成對人類有用的東西，是電腦如此有用的原因。例如，屏幕由數百萬個離散像素組成，每一個像素都太小，人眼無法分辨，但它們結合在一起構成了完整的圖象。通常每個像素都有一個特定的紅色、綠色和藍色的組件來組成它的顯示顏色。當然，這些值可以用數字表示，當然也可以用二進製表示!因此，任何圖象都可以被分割成數百萬個單獨的點，每個點都由表示像素的紅、綠、藍三個值的元組表示。因此，如果給定一長串這樣的數字，格式正確，計算機中的視頻硬件可以將這些數字轉換為電信號，以打開和關閉單個像素，從而顯示圖象。

隨著你的閲讀，我們將建立整個現代計算環境從這個基本的建築塊;如果你願意，從下至上理解整台電腦!

<!--
You may be wondering how a simple number is the basis of all the amazing things a computer can do. Believe it or not, it is! The processor in your computer has a complex but ultimately limited set of instructions it can perform on values such as addition, multiplication, etc. Essentially, each of these instructions is assigned a number so that an entire program (add this to that, multiply by that, divide by this and so on) can be represented by a just a stream of numbers. For example, if the processor knows operation 2 is addition, then 252 could mean "add 5 and 2 and store the output somewhere". The reality is of course much more complicated (see Chapter 3, Computer Architecture) but, in a nutshell, this is what a computer is.

In the days of punch-cards, one could see with their eye the one's and zero's that make up the program stream by looking at the holes present on the card. Of course this moved to being stored via the polarity of small magnetic particles rather quickly (tapes, disks) and onto the point today that we can carry unimaginable amounts of data in our pocket.

Translating these numbers to something useful to humans is what makes a computer so useful. For example, screens are made up of millions of discrete pixels, each too small for the human eye to distinguish but combining to make a complete image. Generally each pixel has a certain red, green and blue component that makes up it's display color. Of course, these values can be represented by numbers, which of course can be represented by binary! Thus any image can be broken up into millions of individual dots, each dot represented by a tuple of three values representing the red, green and blue values for the pixel. Thus given a long string of such numbers, formatted correctly, the video hardware in your computer can convert those numbers to electrical signals to turn on and off individual pixels and hence display an image.

As you read on, we will build up the entire modern computing environment from this basic building block; from the bottom-up if you will!

-->


### 比特和位元組 Bits and Bytes

如上所述，我們基本上可以選擇用一個數字來表示任何東西，這個數字可以轉換成二進制並由計算機操作。例如，為了表示字母表中的所有字母，我們至少需要足夠多的不同組合來表示所有小寫字母、大寫字母、數字和標點符號，以及一些額外的元素。加起來意味著我們可能需要大約80種不同的組合。


如果我們有兩個比特，我們可以表示四個可能的唯一組合(00 01 10 11)。如果有3位，我們可以表示8種不同的組合。一般來說，有n位，我們可以表示 2<sup>n</sup> 個唯一的組合。

8位給我們2<sup>8</sup> = 256的唯一表示，超過了我們的字母表組合。我們稱一組8位的群體為位元組。猜猜 C char 變數有多大?一個位元組。

<!--
As discussed above, we can essentially choose to represent anything by a number, which can be converted to binary and operated on by the computer. For example, to represent all the letters of the alphabet we would need at least enough different combinations to represent all the lower case letters, the upper case letters, numbers and punctuation, plus a few extras. Adding this up means we need probably around 80 different combinations.

If we have two bits, we can represent four possible unique combinations (00 01 10 11). If we have three bits, we can represent 8 different combinations. In general, with n bits we can represent  2<sup>n</sup> unique combinations.

8 bits gives us 2<sup>8</sup> = 256 unique representations, more than enough for our alphabet combinations. We call a group of 8 bits a byte. Guess how big a C char variable is? One byte.

-->

### ASCII

如果一個位元組可以表示0到255的任何值，任何人都可以任意地在字元和數字之間建立映射。例如，顯卡製造商可能會認為1代表a，所以當值1被發送到顯卡時，它會在屏幕上顯示一個大寫的“a”。打印機製造商可能出於某種不明原因決定，1表示小寫的“z”，這意味著顯示和打印相同的東西需要複雜的轉換。


為了避免這種情況發生，發明了美國信息交換標準代碼或ASCII。這是一個7位的代碼，意味著有2<sup>7</sup> 或128個可用代碼。


代碼的範圍分為兩個主要部分;不可打印的和可打印的。可打印字元包括字元(大寫和小寫)、數字和標點符號。不可打印的代碼是用於控制的，可以執行諸如回車、按下終端鈴或根本不代表任何東西的特殊空代碼之類的操作。

127個獨特的字元對於美式英語來說就足夠了，但是當一個人想要表示其他語言中常見的字元時，尤其是亞洲語言中可能有成千上萬個獨特的字元時，就會變得非常不夠了。

為瞭解決這個問題，現代系統正在從 ASCII 轉換到 Unicode, Unicode 可以使用最多 4 個位元組來表示一個字元，從而提供了更多的空間 !

<!--
Given that a byte can represent any of the values 0 through 255, anyone could arbitrarily make up a mapping between characters and numbers. For example, a video card manufacturer could decide that 1 represents A, so when value 1 is sent to the video card it displays a capital 'A' on the screen. A printer manufacturer might decide for some obscure reason that 1 represented a lower-case 'z', meaning that complex conversions would be required to display and print the same thing.

To avoid this happening, the American Standard Code for Information Interchange or ASCII was invented. This is a 7-bit code, meaning there are 27 or 128 available codes.

The range of codes is divided up into two major parts; the non-printable and the printable. Printable characters are things like characters (upper and lower case), numbers and punctuation. Non-printable codes are for control, and do things like make a carriage-return, ring the terminal bell or the special NULL code which represents nothing at all.

127 unique characters is sufficient for American English, but becomes very restrictive when one wants to represent characters common in other languages, especially Asian languages which can have many thousands of unique characters.

To alleviate this, modern systems are moving away from ASCII to Unicode, which can use up to 4 bytes to represent a character, giving much more room!

-->

### 奇偶校驗 Parity

ASCII 是一個 7 位的代碼，只留下一個位元組的空閒。這可以用來實現奇偶校驗，這是一種簡單的錯誤檢查形式。考慮使用打孔卡進行輸入的計算機，其中一個孔表示 1，沒有孔表示 0。任何不小心覆蓋的洞將導致錯誤的值讀取，導致未定義的行為。


奇偶校驗允許對位元組位進行簡單的檢查，以確保它們被正確讀取。我們可以通過使用額外的位作為奇偶校驗位來實現奇偶校驗位或偶校驗位。


在奇校驗中，如果 7 位信息中的1的個數是奇數，奇校驗位就被設置為 1。如果1的個數是偶數，奇校驗位設為0。


通過這種方式，某個位元翻轉 (從 0 變 1 或從 1 變 0) 會導致奇偶校驗錯誤，這是可以檢測到的。


<!--
XXX更多關於錯誤糾正

ASCII, being only a 7-bit code, leaves one bit of the byte spare. This can be used to implement parity which is a simple form of error checking. Consider a computer using punch-cards for input, where a hole represents 1 and no hole represents 0. Any inadvertent covering of a hole will cause an incorrect value to be read, causing undefined behaviour.

Parity allows a simple check of the bits of a byte to ensure they were read correctly. We can implement either odd or even parity by using the extra bit as a parity bit.

In odd parity, if the number of 1's in the 7 bits of information is odd, the parity bit is set, otherwise it is not set. Even parity is the opposite; if the number of 1's is even the parity bit is set to 1.

In this way, the flipping of one bit will case a parity error, which can be detected.

XXX more about error correcting

-->

### 16, 32和64位計算機 (16, 32 and 64 bit computers)

數字不適合位元組; 希望您的銀行餘額不只需要一個位元組來表達! 現代架構至少是 32 位計算機。這意味著它們在處理和讀寫內存時一次處理 4 個位元組。我們把4位元組看作一個單詞; 這類似於一種語言，在這種語言中，字母(比特)組成了一個句子中的單詞，但在計算中，每個單詞都有相同的大小! C int變數的大小是32位。現代的體系結構是64位，這使處理器工作時的大小加倍到8個位元組 (8 bytes = 64 bits)。

<!--
Numbers do not fit into bytes; hopefully your bank balance in dollars will need more range than can fit into one byte! Modern architectures are at least 32 bit computers. This means they work with 4 bytes at a time when processing and reading or writing to memory. We refer to 4 bytes as a word; this is analogous to language where letters (bits) make up words in a sentence, except in computing every word has the same size! The size of a C int variable is 32 bits. Modern architectures are 64 bits, which doubles the size the processor works with to 8 bytes.

-->

### Kilo, Mega and Giga Bytes

計算機處理大量位元組; 這就是他們強大的原因! 我們需要一種談論大量位元組的方法，一種自然的方法是使用 “國際單位制”(SI) ，就像大多數其他科學領域所使用的那樣。 例如，千克是指10<sup>3</sup>或1000單位，千克是1000克。


以 10 為基數的 1000 是一個不錯的整數，但在二進制中它是 1111101000 ，這不是一個特別的 “整數”。然而，1024 (或2<sup>10</sup>) 是一個整數 —— (10000000000)，恰好與“kilo” (1000，而不是1024) 的基數10非常接近。因此，1024 位元組自然被稱為千位元組。下一個 SI 單元是 “mega”，代表10<sup>6</sup>個單位，首碼繼續向上10<sup>3</sup>個單位(相當於寫大數時通常的三位數分組)。實際情況是，2<sup>20</sup> 再次接近以10為基數的SI對mega的定義; 1048576 而不是1000000。以10的冪來增加基數2的單位在功能上仍然接近 SI 基數 10 的值，儘管每個增加的因素都略微偏離SI基數的含義。因此，SI 基-10單位是“足夠接近”的，已經成為常用的以 2 為基數的值。

<!--
Computers deal with a lot of bytes; that's what makes them so powerful! We need a way to talk about large numbers of bytes, and a natural way is to use the "International System of Units" (SI) prefixes as used in most other scientific areas. So for example, kilo refers to <sup>3</sup> or 1000 units, as in a kilogram has 1000 grams.

1000 is a nice round number in base 10, but in binary it is 1111101000 which is not a particularly "round" number. However, 1024 (or 2<sup>10</sup>) is a round number — (10000000000 — and happens to be quite close to the base 10 meaning value of "kilo" (1000 as opposed to 1024). Thus 1024 bytes naturally became known as a kilobyte. The next SI unit is "mega" for 10<sup>6</sup> and the prefixes continue upwards by 10<sup>3</sup> (corresponding to the usual grouping of three digits when writing large numbers). As it happens, 2<sup>20</sup> is again close to the SI base 10 definition for mega; 1048576 as opposed to 1000000. Increasing the base 2 units by powers of 10 remains functionally close to the SI base 10 value, although each increasing factor diverges slightly further from the base SI meaning. Thus the SI base-10 units are "close enough" and have become the commonly used for base 2 values.

-->

> 表2.4。基數2和10與位元組相關的因數 (Base 2 and 10 factors related to bytes)

名字 Name |	二的次方 Base 2 Factor | Bytes |	接近的十進位 Close Base 10 Factor | Base 10 bytes
-----|----------------|-------|----------------------|------------------
1 Kilobyte |	2<sup>10<sup> |	1,024 |	10<sup>3<sup> |	1,000
1 Megabyte |	2<sup>20<sup> |	1,048,576 |	10<sup>6<sup> |	1,000,000
1 Gigabyte |	2<sup>30<sup> |	1,073,741,824 |	10<sup>9<sup> |	1,000,000,000
1 Terabyte |	2<sup>40<sup> |	1,099,511,627,776 |	10<sup>12<sup> |	1,000,000,000,000
1 Petabyte |	2<sup>50<sup> |	1,125,899,906,842,624 |	10<sup>15<sup> |	1,000,000,000,000,000
1 Exabyte |	2<sup>60<sup> |	1,152,921,504,606,846,976 |	10<sup>18<sup> |	1,000,000,000,000,000,000

> 以2為基數和以10為基數進行比較
> SI units compared in base 2 and base 10

將基本的2因素提交到內存中以幫助快速關聯比特數和“人類”大小之間的關係，這是非常有用的。例如，我們可以通過記錄2<sup>2</sup>(4)+ 230的重新組合，快速計算出一台32位計算機最多可以處理4g內存。一個64位的值同樣可以處理16 Exabyte (2<sup>4</sup> + 2<sup>60</sup>);你可能會對計算出這個數有多大感興趣。要想知道這個數字有多大，計算一下如果每秒遞增一次，要數到 2<sup>64</sup> 需要多長時間。

<!--
It can be very useful to commit the base 2 factors to memory as an aid to quickly correlate the relationship between number-of-bits and "human" sizes. For example, we can quickly calculate that a 32 bit computer can address up to four gigabytes of memory by noting the recombination of 2<sup>2</sup> (4) + 2<sup>30</sup>. A 64-bit value could similarly address up to 16 exabytes (2<sup>4</sup> + 2<sup>60</sup>); you might be interested in working out just how big a number this is. To get a feel for how big that number is, calculate how long it would take to count to 2<sup>64</sup> if you incremented once per second.

-->

### Kilo, Mega and Giga Bits

除了與二進制和以10為基數的SI單位的重載有關的混淆之外，容量通常用位而不是位元組來表示。通常這發生在談論網絡或存儲設備時;您可能已經注意到您的ADSL連接被描述為大約1500千比特/秒。計算簡單;乘以1000(千克)，除以8得到位元組，然後是1024得到千位元組(1500千比特/秒=183千位元組每秒)。

SI標準化主體已經識別了這些雙重用途，併為二進制用法指定了唯一的首碼。在標準 1024 位元組下是一個 kibibyte，是千位元組的簡稱(簡稱KiB)。其他首碼有相似的首碼(例如 Mebibyte, MiB)。傳統習慣在很大程度上避免了使用這些術語，但你可能在某些文學作品中看到它們。

<!--
Apart from the confusion related to the overloading of SI units between binary and base 10, capacities will often be quoted in terms of bits rather than bytes. Generally this happens when talking about networking or storage devices; you may have noticed that your ADSL connection is described as something like 1500 kilobits/second. The calculation is simple; multiply by 1000 (for the kilo), divide by 8 to get bytes and then 1024 to get kilobytes (so 1500 kilobits/s=183 kilobytes per second).

The SI standardisation body has recognised these dual uses and has specified unique prefixes for binary usage. Under the standard 1024 bytes is a kibibyte, short for kilo binary byte (shortened to KiB). The other prefixes have a similar prefix (Mebibyte, MiB, for example). Tradition largely prevents use of these terms, but you may seem them in some literature.

-->

### 轉化 Conversion

最簡單的轉換方式是使用電腦，畢竟，這是他們擅長的!但是，知道如何手工進行轉換通常很有用。

基之間最簡單的轉換方法是重複除法。要進行轉換，要反覆地把商除以基數，直到商為零，在每一步都要記下餘數。然後，把餘數反過來寫，從下面開始，每次都在右邊加上。舉例說明;因為我們要轉換成二進制，所以我們用 2 為底。

<!--
The easiest way to convert between bases is to use a computer, after all, that's what they're good at! However, it is often useful to know how to do conversions by hand.

The easiest method to convert between bases is repeated division. To convert, repeatedly divide the quotient by the base, until the quotient is zero, making note of the remainders at each step. Then, write the remainders in reverse, starting at the bottom and appending to the right each time. An example should illustrate; since we are converting to binary we use a base of 2.

-->

> 表2.5。將203轉換為二進制 (Convert 203 to binary)

Quotient | 　	  |	Remainder	 | 　
---------|--------|----------------------|-----------
203<sub>10</sub> ÷ 2 = | 101 |	1 |	  |
101<sub>10</sub> ÷ 2 = | 50 |	1 |	↑ |
50<sub>10</sub> ÷ 2 = |	25 |	0 |	↑ |
25<sub>10</sub> ÷ 2 = |	12 |	1 |	↑ |
12<sub>10</sub> ÷ 2 = |	6 |	0 |	↑ |
6<sub>10</sub> ÷ 2 = |	3 |	0 |	↑ |
3<sub>10</sub> ÷ 2 = |	1 |	1 |	↑ |
1<sub>10</sub> ÷ 2 = |	0 |	1 |	↑ |

### 布爾操作 Boolean Operations

喬治·布爾是一位數學家，他發現了一門叫做布爾代數的數學領域。當他在19世紀中期做出發現時，他的數學是所有計算機科學的基礎。布爾代數是一個範圍很廣的話題，我們在這裡只給出最基本的開始。


布爾操作只是接受一個特定的輸入並根據規則產生一個特定的輸出。例如，最簡單的布爾運算，並不是簡單地反轉輸入操作數的值。其他操作數通常接受兩個輸入，併產生一個輸出。


計算機科學中使用的基本布爾運算很容易記住，下面列出。我們用真值表來表示它們;它們只是顯示所有可能的輸入和輸出。真這個詞簡單地反映了二進制中的 1。

<!--
George Boole was a mathematician who discovered a whole area of mathematics called Boolean Algebra. Whilst he made his discoveries in the mid 1800's, his mathematics are the fundamentals of all computer science. Boolean algebra is a wide ranging topic, we present here only the bare minimum to get you started.

Boolean operations simply take a particular input and produce a particular output following a rule. For example, the simplest boolean operation, not simply inverts the value of the input operand. Other operands usually take two inputs, and produce a single output.

The fundamental Boolean operations used in computer science are easy to remember and listed below. We represent them below with truth tables; they simply show all possible inputs and outputs. The term true simply reflects 1 in binary.

-->

### Not

通常用!來表示，而不是簡單地求倒數，所以 0 變成 1, 1 變成 0

<!--
Usually represented by !, not simply inverts the value, so 0 becomes 1 and 1 becomes 0

-->

> 表2.6。 Not 的真值表

Input|	Output
------|--------
1|	0
0|	1

### And

要記住和運算是如何工作的，可以把它想像成“如果一個輸入和另一個輸入是正確的，那麼結果就是正確的。

<!--
To remember how the and operation works think of it as "if one input and the other are true, result is true

-->

> 表2.7。 AND 的真值表

Input 1	| Input 2 | Output
--------|---------|----------
0 | 	0 | 	0
1 | 	0 | 	0
0 | 	1 | 	0
1 | 	1 | 	1

### Or

要記住 or 操作是如何工作的，請將其視為 “如果一個輸入或另一個輸入為真，則結果為真”

<!--
To remember how the or operation works think of it as "if one input or the other input is true, the result is true

-->

> 表2.8。 OR 的真值表


Input 1 | Input 2 | Output
--------|---------|----------
0 | 	0 | 	0
1 | 	0 | 	1
0 | 	1 | 	1
1 | 	1 | 	1

### Exclusive Or (xor)

如果輸入中的一個(且僅一個)為真，則用Exclusive or(異或)表示，它是輸出為真或為真的特殊情況。這個操作可以令人驚訝地執行許多有趣的技巧，但是在內核中不會看到很多。

<!--
Exclusive or, written as xor is a special case of or where the output is true if one, and only one, of the inputs is true. This operation can surprisingly do many interesting tricks, but you will not see a lot of it in the kernel.
-->

> 表2.9。 Xor 的真值表

Input 1 | Input 2 | Output
--|---|--
0 | 0 | 0
1 | 0 | 1
0 | 1 | 1
1 | 1 | 0

## 計算機如何使用布爾運算 How computers use boolean operations

信不信由你，基本上你的電腦所做的一切都回到了上面的操作。例如，半加法器是一種由布爾運算組成的電路，它可以把位加在一起(它被稱為半加法器，因為它不處理進位)。將一半以上的加法器放在一起，您就可以開始構建可以將長二進制數相加的東西。加上一些外部存儲器，你就有一台電腦了。


在電子技術上，布爾運算是在晶體管製造的門中實現的。這就是為什麼你可能聽說過晶體管計數和摩爾定律。晶體管越多，門越多，可以加在一起的東西就越多。為了創造現代計算機，有很多門，很多晶體管。一些最新的安騰處理器擁有大約4.6億個晶體管。

<!--
Believe it or not, essentially everything your computer does comes back to the above operations. For example, the half adder is a type of circuit made up from boolean operations that can add bits together (it is called a half adder because it does not handle carry bits). Put more half adders together, and you will start to build something that can add together long binary numbers. Add some external memory, and you have a computer.

Electronically, the boolean operations are implemented in gates made by transistors. This is why you might have heard about transistor counts and things like Moore's Law. The more transistors, the more gates, the more things you can add together. To create the modern computer, there are an awful lot of gates, and an awful lot of transistors. Some of the latest Itanium processors have around 460 million transistors.

-->

### 在C語言中使用二進制 Working with binary in C

在C語言中，我們有對所有上述操作的直接接口。下表描述了操作符

<!--
In C we have a direct interface to all of the above operations. The following table describes the operators

-->

> 表2.10。C 中的布爾運算


操作 Operation | C 的符號 Usage in C
----|----
not | !
and | &
or | |
xor | ^

我們對變數使用這些操作來修改變數中的位。在我們看到這個例子之前，首先我們必須轉移到描述十六進製表示法。

<!--
We use these operations on variables to modify the bits within the variable. Before we see examples of this, first we must divert to describe hexadecimal notation.

-->

### 十六進制 Hexadecimal

十六進制是指以16為基數的數字系統。我們在計算機科學中使用這個只有一個原因，它使人們更容易思考二進制數字。計算機只能處理二進制和十六進制的數據，這對我們人類使用計算機來說是一條捷徑。


為什麼以16為底?最自然的選擇是以10為基數，因為我們習慣於從我們每天的數字系統中以10為基數來思考。但是以10為底不能很好地表示二進制——要表示二進制中的10個不同的元素，我們需要4位。然而，4位，給了我們16種可能的組合。因此，我們可以採取非常棘手的方法，試圖在10和二進制之間進行轉換，或者採取簡單的方法，構建一個16進制的數字系統——十六進制!


十六進制使用的是以10為基數的標準數字，但加了一個B C D E F，表示 10 11 12 13 14 15(注意，我們從0開始)。


傳統上，任何時候你看到一個以 0x 為首碼的數字，這將表示一個十六進制數。


如前所述，要在二進制中表示16種不同的模式，我們需要4位。因此，每個十六進制數恰好代表4位。你應該把背下這張表當作一種練習。

<!--
Hexadecimal refers to a base 16 number system. We use this in computer science for only one reason, it makes it easy for humans to think about binary numbers. Computers only ever deal in binary and hexadecimal is simply a shortcut for us humans trying to work with the computer.

So why base 16? Well, the most natural choice is base 10, since we are used to thinking in base 10 from our every day number system. But base 10 does not work well with binary -- to represent 10 different elements in binary, we need four bits. Four bits, however, gives us sixteen possible combinations. So we can either take the very tricky road of trying to convert between base 10 and binary, or take the easy road and make up a base 16 number system -- hexadecimal!

Hexadecimal uses the standard base 10 numerals, but adds A B C D E F which refer to 10 11 12 13 14 15 (n.b. we start from zero).

Traditionally, any time you see a number prefixed by 0x this will denote a hexadecimal number.

As mentioned, to represent 16 different patterns in binary, we would need exactly four bits. Therefore, each hexadecimal numeral represents exactly four bits. You should consider it an exercise to learn the following table off by heart.

-->

> 表 2.11. 十六進制, 二進制 and 十進制

十六進制 Hexadecimal | 二進制 Binary | 十進制 Decimal
--|----------|------- 
0 | 	0000 | 	0
1 | 	0001 | 	1
2 | 	0010 | 	2
3 | 	0011 | 	3
4 | 	0100 | 	4
5 | 	0101 | 	5
6 | 	0110 | 	6
7 | 	0111 | 	7
8 | 	1000 | 	8
9 | 	1001 | 	9
A | 	1010 | 	10
B | 	1011 | 	11
C | 	1100 | 	12
D | 	1101 | 	13
E | 	1110 | 	14
F | 	1111 | 	15

當然沒有理由不繼續採用這種模式(例如,分配 G 字母給 16),但16個值是一個很好的權衡之間變幻莫測的人類記憶和計算機使用的位數(偶爾你也會看到基地8使用,例如UNIX下的文件權限)。我們只是用更多的數字表示更大的比特數。例如，一個16位的變數可以用0xAB12表示，要找到二進制的變數，只需將每個單獨的數字轉換成表，並將它們連接在一起(因此0xAB12最終是16位二進制數字1010101100010010)。我們可以使用反向將二進制轉換回十六進制。


我們也可以使用相同的重複除法方案來改變一個數字的基數。例如，在十六進制中找到203

<!--
Of course there is no reason not to continue the pattern (say, assign G to the value 16), but 16 values is an excellent trade off between the vagaries of human memory and the number of bits used by a computer (occasionally you will also see base 8 used, for example for file permissions under UNIX). We simply represent larger numbers of bits with more numerals. For example, a sixteen bit variable can be represented by 0xAB12, and to find it in binary simply take each individual numeral, convert it as per the table and join them all together (so 0xAB12 ends up as the 16-bit binary number 1010101100010010). We can use the reverse to convert from binary back to hexadecimal.

We can also use the same repeated division scheme to change the base of a number. For example, to find 203 in hexadecimal

-->

> 表2.12。將 203 轉換為十六進制 (Convert 203 to hexadecimal)

商 Quotient     |     | 餘數 Remainder | 
-------------|-----|------------------ 
203<sub>10</sub> ÷ 16 = | 	12 | 11 (0xB) | 
12<sub>10</sub> ÷ 16 =  | 0   | 12 (0xC) | ↑

因此，十六進制中的203是0xCB。

<!--
Hence 203 in hexadecimal is 0xCB.

-->

### 實際意義 Practical Implications

### 在代碼中使用二進制代碼 Use of binary in code

雖然二進制是每台計算機的基礎語言，但是用高級語言編寫計算機程序而不知道它的基礎原理是完全可行的。然而，對於低級代碼，我們感興趣的是一些基本的二進制原則被反覆使用。

<!--
Whilst binary is the underlying language of every computer, it is entirely practical to program a computer in high level languages without knowing the first thing about it. However, for the low level code we are interested in a few fundamental binary principles are used repeatedly.

-->

### 屏蔽和旗幟 Masking and Flags

### 屏蔽 Masking

在低級代碼中，保持結構和變數儘可能高效通常是很重要的。在某些情況下，這可能涉及到將兩個(通常相關的)變數有效地打包為一個變數。


記住，每個位代表兩種狀態，所以如果我們知道一個變數隻有，比如說，16種可能的狀態，它可以用4位表示(即24=16個唯一值)。但是我們可以在C中聲明的最小類型是8位(一個char)，所以我們可以浪費4位，或者找到一些方法來使用這些剩餘的位。

我們可以通過屏蔽 Mask 的過程很容易地做到這一點。要記住邏輯操作的規則，應該很清楚如何提取值。

這個過程如下圖所示。我們感興趣的是下面的4位，因此將蒙版設置為這些位設置為1。由於邏輯和操作只會在兩個位都為1的情況下設置位，因此屏蔽的這些位設置為0實際上會隱藏我們不感興趣的位。

<!--
In low level code, it is often important to keep your structures and variables as space efficient as possible. In some cases, this can involve effectively packing two (generally related) variables into one.

Remember each bit represents two states, so if we know a variable only has, say, 16 possible states it can be represented by 4 bits (i.e. 24=16 unique values). But the smallest type we can declare in C is 8 bits (a char), so we can either waste four bits, or find some way to use those left over bits.

We can easily do this by the process of masking. Remembering the rules of the logical operations, it should become clear how the values are extracted.

The process is illustrated in the figure below. We are interested in the lower four bits, so set our mask to have these bits set to 1. Since the logical and operation will only set the bit if both bits are 1, those bits of the mask set to 0 effectively hide the bits we are not interested in.

-->

> 圖2.1。屏蔽 (Masking)

![Figure 2.1. Masking](http://www.bottomupcs.com/chapter01/figures/masking.png)


為了得到頂部(藍色)4位，我們將反轉 Mask。當我們真正想要0x09時，您會注意到這給出了0x90的結果。為了把比特放到正確的位置，我們使用了正確的移位操作。

設置位需要邏輯或操作。但是，我們使用的不是1，而是0。您應該繪製一個類似於上面圖的圖表，並使用邏輯或操作來設置位。

<!--
To get the top (blue) four bits, we would invert the mask. You will note this gives a result of 0x90 when really we want a value of 0x09. To get the bits into the right position we use the right shift operation.

Setting the bits requires the logical or operation. However, rather than using 1's as the mask, we use 0's. You should draw a diagram similar to the above figure and work through setting bits with the logical or operation.

-->

### 旗幟 Flags

通常，程序會有大量的變數，這些變數作為標記存在於某些條件中。例如，狀態機是一種通過多個不同狀態進行轉換的算法，但一次只能在一個狀態中進行轉換。假設它有8種不同的狀態;我們可以很容易地聲明8個不同的變數，每個變數對應一個狀態。但在許多情況下，最好是聲明一個8位變數，並將每個位分配給標誌一個特定狀態。

標誌是屏蔽 Mask 的一種特殊情況，但是每個位代表一個特定的布爾狀態(打開或關閉)。一個n位變數可以容納n個不同的標誌。關於使用標誌的典型示例，請參閱下面的代碼示例——您將經常看到這種基本代碼的變體。

<!--
Often a program will have a large number of variables that only exist as flags to some condition. For example, a state machine is an algorithm that transitions through a number of different states but may only be in one at a time. Say it has 8 different states; we could easily declare 8 different variables, one for each state. But in many cases it is better to declare one 8 bit variable and assign each bit to flag flag a particular state.

Flags are a special case of masking, but each bit represents a particular boolean state (on or off). An n bit variable can hold n different flags. See the code example below for a typical example of using flags -- you will see variations on this basic code very often.

-->

> 例2.1。使用標誌 (Using flags)

```c
#include <stdio.h>

/*
 *  define all 8 possible flags for an 8 bit variable
 *      name  hex     binary
 */
#define FLAG1 0x01 /* 00000001 */
#define FLAG2 0x02 /* 00000010 */
#define FLAG3 0x04 /* 00000100 */
#define FLAG4 0x08 /* 00001000 */
/* ... and so on */
#define FLAG8 0x80 /* 10000000 */

int main(int argc, char *argv[])
{
	char flags = 0; /* an 8 bit variable */

	/* set flags with a logical or */
	flags = flags | FLAG1; /* set flag 1 */
	flags = flags | FLAG3; /* set flag 3
	/* check flags with a logical and.  If the flag is set (1)
	 * then the logical and will return 1, causing the if
	 * condition to be true. */
	if (flags & FLAG1)
		printf("FLAG1 set!\n");

	/* this of course will be untrue. */
	if (flags & FLAG8)
		printf("FLAG8 set!\n");

	/* check multiple flags by using a logical or
	 * this will pass as FLAG1 is set */
	if (flags & (FLAG1|FLAG4))
		printf("FLAG1 or FLAG4 set!\n");

	return 0;
}
```

## 類型和數字表示 Types and Number Representation

### C標準 (C Standards)

雖然有一點分歧，但是瞭解一點關於C語言的歷史是很重要的。

C是系統編程世界的通用語言。每個操作系統及其相關的常用系統庫都是用C語言編寫的，每個系統都提供一個C編譯器。為了防止語言在這些系統之間產生分歧，因而造成系統移植時必須進行大量的更改，因此 C 語言設定了嚴格的標準。

這個標準的官方名稱是ISO/IEC 989:1999(E)，但通常簡稱為C99。該標準由國際標準化組織(ISO)維護，完整標準可在線購買。較老的標準版本，如C89(1989年發佈的C99的前身)和ANSI C不再常用，並被包含在最新的標準中。標準文檔是非常技術性的，詳細描述了語言的每個部分。例如，它解釋了語法(在Backus Naur形式中)、標準 #define 以及操作的行為方式。

同樣重要的是要注意C標準沒有定義什麼。最重要的是，標準需要適合當前和未來的所有硬體結構。因此，它不需要依賴於硬體結構。C標準和底層架構之間的“粘合劑”是應用程序二進制接口(或ABI)，我們將在下面討論。在一些地方，標準會提到一個特定的操作或構造有一個不確定的或依賴於實現的結果。顯然，如果程序員要編寫可移植的代碼，就不能依賴於這些不確定的項目。

<!--
Although a slight divergence, it is important to understand a bit of history about the C language.

C is the common languge of the systems programming world. Every operating system and its associated system libraries in common use is written in C, and every system provides a C compiler. To stop the language diverging across each of these systems where each would be sure to make numerous incompatible changes, a strict standard has been written for the language.

Officially this standard is known as ISO/IEC 9899:1999(E), but is more commonly referred to by its shortened name C99. The standard is maintained by the International Standards Organisation (ISO) and the full standard is available for purchase online. Older standards versions such as C89 (the predecessor to C99 released in 1989) and ANSI C are no longer in common usage and are encompassed within the latest standard. The standard documentation is very technical, and details most every part of the language. For example it explains the syntax (in Backus Naur form), standard #define values and how operations should behave.

It is also important to note what the C standards does not define. Most importantly the standard needs to be appropriate for every architecture, both present and future. Consequently it takes care not to define areas that are architecture dependent. The "glue" between the C standard and the underlying architecture is the Application Binary Interface (or ABI) which we discuss below. In several places the standard will mention that a particular operation or construct has an unspecified or implementation dependent result. Obviously the programmer can not depend on these outcomes if they are to write portable code.

-->

### GNU C

GNU C編譯器(通常稱為gcc)幾乎完全實現了C99標準。然而，它也實現了對標準的一系列擴展，程序員通常會使用這些擴展來獲得額外的功能，而代價是犧牲了對另一個編譯器的可移植性。這些擴展通常與非常低級的代碼相關，在系統編程領域更為常見;在這個領域中最常用的擴展是內聯彙編代碼。程序員應該閲讀gcc文檔，並瞭解何時使用與標準不同的特性。

可以指示gcc嚴格遵守標準(例如-std=c99標誌)，並在執行某些不屬於標準的任務時發出警告或創建錯誤。如果您需要確保您的代碼可以輕鬆地移動到另一個編譯器，那麼設定 c99 選項顯然是合適的。

<!--
The GNU C Compiler, more commonly referred to as gcc, almost completely implements the C99 standard. However it also implements a range of extensions to the standard which programmers will often use to gain extra functionality, at the expense of portability to another compiler. These extensions are usually related to very low level code and are much more common in the system programming field; the most common extension being used in this area being inline assembly code. Programmers should read the gcc documentation and understand when they may be using features that diverge from the standard.

gcc can be directed to adhere strictly to the standard (the -std=c99 flag for example) and warn or create an error when certain things are done that are not in the standard. This is obviously appropriate if you need to ensure that you can move your code easily to another compiler.

-->

### 類型 Types

作為程序員，我們熟悉使用變數來表示存儲值的內存區域。在類型化語言(如C)中，每個變數都必須用類型聲明。類型告訴編譯器我們希望在變數中存儲什麼;然後，編譯器可以為這種用法分配足夠的空間，並檢查程序員是否違反類型規則。在下面的示例中，我們將看到為某些常見類型的變數分配的空間示例。

<!--
As programmers, we are familiar with using variables to represent an area of memory to hold a value. In a typed language, such as C, every variable must be declared with a type. The type tells the compiler about what we expect to store in a variable; the compiler can then both allocate sufficient space for this usage and check that the programmer does not violate the rules of the type. In the example below, we see an example of the space allocated for some common types of variables.

-->

> 圖2.2。類型 (Types)

![](http://www.bottomupcs.com/chapter01/figures/types.png)


C99標準故意只提到為c定義的每個類型的最小可能大小，這是因為在不同的處理器體系結構和操作系統中，類型的最佳大小可能相差很大。

為了完全安全，程序員永遠不需要假設任何變數的大小，然而一個正常運行的系統顯然需要就系統中將要使用的大小類型達成一致。每個體系結構和操作系統都符合應用程序二進制接口或ABI。系統的ABI填補了C標準與底層硬件和操作系統需求之間的細節。ABI是為特定處理器和操作系統的組合而編寫的。

<!--
The C99 standard purposely only mentions the smallest possible size of each of the types defined for C. This is because across different processor architectures and operating systems the best size for types can be wildly different.

To be completely safe programmers need to never assume the size of any of their variables, however a functioning system obviously needs agreements on what sizes types are going to be used in the system. Each architecture and operating system conforms to an Application Binary Interface or ABI. The ABI for a system fills in the details between the C standard and the requirements of the underlying hardware and operating system. An ABI is written for a specific processor and operating system combination.

-->

> 表2.13。標準整數類型和大小 (Standard Integer Types and Sizes)

Type 型態 | C99最小尺寸(比特) C99 minimum size (bits) | 普通尺寸(32位架構) Common size (32 bit architecture)
-----|---|-------
char | 8 | 8
short | 16 | 16
int | 16 | 32
long | 32 | 32
long long | 64 | 64
Pointers | 依賴於實現的 Implementation dependent | 32

在上面，我們可以看到與標準的唯一不同之處是int通常是一個32位的數量，這是C99所要求的嚴格最小16位大小的兩倍。


指針實際上只是一個地址(也就是說，它們的值是一個地址，因此“點”在內存中的其他地方)，因此指針需要足夠大的空間來處理系統中的任何內存。

<!--
Above we can see the only divergence from the standard is that int is commonly a 32 bit quantity, which is twice the strict minimum 16 bit size that the C99 requires.

Pointers are really just an address (i.e. their value is an address and thus "points" somewhere else in memory) therefore a pointer needs to be sufficient in size to be able to address any memory in the system.

-->

### 64 bit

引起混亂的一個領域是64位計算的引入。這意味著處理器可以處理長度為64位的地址(特別是寄存器為64位寬;這是我們在第三章“計算機架構”中討論的主題。


這首先意味著所有指針都必須是64位寬的，這樣它們就可以表示系統中任何可能的地址。然而，系統實現者必須決定其他類型的大小。兩種常見的模型被廣泛使用，如下所示。

<!--
One area that causes confusion is the introduction of 64 bit computing. This means that the processor can handle addresses 64 bits in length (specifically the registers are 64 bits wide; a topic we discuss in Chapter 3, Computer Architecture).

This firstly means that all pointers are required to be a 64 bits wide so they can represent any possible address in the system. However, system implementers must then make decisions about the size of the other types. Two common models are widely used, as shown below.

-->

> 表2.14。標準標量類型和大小 (Standard Scalar Types and Sizes)

型態 Type | C99最小尺寸(比特) C99 minimum size (bits) | 普通尺寸 Common size (LP64) | 普通尺寸 Common size (Windows)
-----|---|---|---
char | 8 | 8 | 8
short | 16 | 16 | 16
int | 16 | 32 | 32
long | 32 | 64 | 32
long long | 64 | 64 | 64
Pointers | Implementation dependent | 64 | 64

可以看到，在LP64(長指針64)模型中，長值 long 被定義為64位寬。這與我們之前展示的32位模型不同。LP64模型在UNIX系統中廣泛使用。


在另一個模型中，long保持32位的值。這將保持與32代碼的最大兼容性。這個模型在64位窗口中使用。


在這兩種模型中，int的大小都沒有增加到64位，這是有充分理由的。考慮到如果int的大小增加到64位，程序員就無法獲得32位的變數。唯一可能的是重新定義 short 為更大的32位類型。


64位變數非常大，通常不需要表示很多變數。例如，循環的重複次數很少超過32位變數的範圍 (4294967296次!) 。圖象通常用8位表示一個紅、綠、藍值，另外8位表示額外的(alpha通道)信息;總共32位。因此，在許多情況下，使用64位變數將浪費至少前32位(如果不是更多的話)。不僅如此，一個整數數組的大小現在也翻了一番。這意味著程序會占用更多的系統內存(從而占用更多的緩存;詳細討論在第三章，計算機架構)沒有真正的改進。出於同樣的原因，Windows選擇將其長值保持為32位;由於很多Windows API最初是為32位系統上使用長變數而編寫的，因此不需要額外的位，這節省了系統中大量的浪費空間，而無需重寫所有的API。


如果我們考慮建議的替代方案，其中short被重新定義為一個32位變數;在64位系統上工作的程序員可以用它來處理他們知道有限制的變數。然而，當回到32位系統時，它們相同的短變數現在只會是16位長，這個值實際上會溢出(65536)。


通過讓程序員在知道需要時請求更大的變數，可以在考慮可移植性和浪費二進制文件空間方面取得平衡。

<!--
You can see that in the LP64 (long-pointer 64) model long values are defined to be 64 bits wide. This is different to the 32 bit model we showed previously. The LP64 model is widely used on UNIX systems.

In the other model, long remains a 32 bit value. This maintains maximum compatibility with 32 code. This model is in use with 64 bit Windows.

There are good reasons why the size of int was not increased to 64 bits in either model. Consider that if the size of int is increased to 64 bits you leave programmers no way to obtain a 32 bit variable. The only possibly is redefining shorts to be a larger 32 bit type.

A 64 bit variable is so large that it is not generally required to represent many variables. For example, loops very rarely repeat more times than would fit in a 32 bit variable (4294967296 times!). Images usually are usually represented with 8 bits for each of a red, green and blue value and an extra 8 bits for extra (alpha channel) information; a total of 32 bits. Consequently for many cases, using a 64 bit variable will be wasting at least the top 32 bits (if not more). Not only this, but the size of an integer array has now doubled too. This means programs take up more system memory (and thus more cache; discussed in detail in Chapter 3, Computer Architecture) for no real improvement. For the same reason Windows elected to keep their long values as 32 bits; since much of the Windows API was originally written to use long variables on a 32 bit system and hence does not require the extra bits this saves considerable wasted space in the system without having to re-write all the API.

If we consider the proposed alternative where short was redefined to be a 32 bit variable; programmers working on a 64 bit system could use it for variables they know are bounded to smaller values. However, when moving back to a 32 bit system their same short variable would now be only 16 bits long, a value which is much more realistically overflowed (65536).

By making a programmer request larger variables when they know they will be needed strikes a balance with respect to portability concerns and wasting space in binaries.

-->

### 類型限定符 Type qualifiers

C標準還討論了一些變數類型的限定符。例如，const意味著變數永遠不會從其原始值修改，volatile向編譯器表明這個值可能會在程序執行流之外更改 (揮發性的)，因此編譯器必須小心，不要以任何方式重新排序對它的訪問。


有符號和無符號可能是兩個最重要的限定詞;題目說一個變數是否可以取負值。我們將在下面更詳細地研究這個問題。


限定符都是為了向編譯器傳遞關於變數如何使用的額外信息。這意味著兩件事;編譯器可以檢查你是否違反了你自己的規則(例如，寫一個常量值)，並且它可以根據額外的知識進行優化(在後面的章節中討論)。

<!--
The C standard also talks about some qualifiers for variable types. For example const means that a variable will never be modified from its original value and volatile suggests to the compiler that this value might change outside program execution flow so the compiler must be careful not to re-order access to it in any way.

signed and unsigned are probably the two most important qualifiers; and they say if a variable can take on a negative value or not. We examine this in more detail below.

Qualifiers are all intended to pass extra information about how the variable will be used to the compiler. This means two things; the compiler can check if you are violating your own rules (e.g. writing to a const value) and it can make optimisations based upon the extra knowledge (examined in later chapters).

-->

### 標準類型 Standard Types

C99意識到，所有這些規則、大小和可移植性問題都會很快變得非常混亂。為了提供幫助，它提供了一系列特殊類型，可以指定變數的確切屬性。這些類型定義在 stdint.h 檔案中以形式 qtypes_t 的方式存在，其中 q 是一個限定符，類型是基本類型，s是位的寬度，_t是一個擴展，所以你知道你在使用C99定義的類型。


例如 uint8_t 是一個 8 比特寬的無符號整數。定義了許多其他類型;完整的列表在 c99 表頭文件 .h 中。[3]


通過將這些類型映射到目標系統上適當大小的類型，由實現C99標準的系統為您提供這些類型;在Linux上，這些頭文件由系統庫提供。

<!--
C99 realises that all these rules, sizes and portability concerns can become very confusing very quickly. To help, it provides a series of special types which can specify the exact properties of a variable. These are defined in `<stdint.h>` and have the form qtypes_t where q is a qualifier, type is the base type, s is the width in bits and _t is an extension so you know you are using the C99 defined types.

So for example uint8_t is an unsigned integer exactly 8 bits wide. Many other types are defined; the complete list is detailed in C99 17.8 or (more cryptically) in the header file. [3]

It is up to the system implementing the C99 standard to provide these types for you by mapping them to appropriate sized types on the target system; on Linux these headers are provided by the system libraries.

-->

### 類型的作用 Types in action

下面的 “示例2.2 類型不匹配時的警告示例” 中，我們將看到一個示例，它說明了類型如何對變數的有效操作施加限制，以及當變數以不正確的方式使用時，編譯器如何使用此信息來警告。在這段代碼中，我們首先給char變數賦一個整數值。因為char變數更小，所以我們釋放了整數的正確值。再往下，我們嘗試將一個指向char類型的指針分配給指定為整數的內存。這個操作可以完成;但這並不安全。第一個示例在32位的奔騰計算機上運行，並返回正確的值。然而，正如第二個示例所示，在64位的安騰計算機上，指針是64位(8位元組)長，而整數隻有4位元組長。顯然，8個位元組不能容納4個位元組!我們可以嘗試通過在賦值前轉換值來“愚弄”編譯器;注意，在本例中，我們通過執行這種強制轉換而忽略編譯器警告，這是對我們自己的一種打擊，因為較小的變數不能保存指針的所有信息，我們最終會得到一個無效的地址。

<!--
Below in Example 2.2, “Example of warnings when types are not matched” we see an example of how types place restrictions on what operations are valid for a variable, and how the compiler can use this information to warn when variables are used in an incorrect fashion. In this code, we firstly assign an integer value into a char variable. Since the char variable is smaller, we loose the correct value of the integer. Further down, we attempt to assign a pointer to a char to memory we designated as an integer. This operation can be done; but it is not safe. The first example is run on a 32-bit Pentium machine, and the correct value is returned. However, as shown in the second example, on a 64-bit Itanium machine a pointer is 64 bits (8 bytes) long, but an integer is only 4 bytes long. Clearly, 8 bytes can not fit into 4! We can attempt to "fool" the compiler by casting the value before assigning it; note that in this case we have shot ourselves in the foot by doing this cast and ignoring the compiler warning since the smaller variable can not hold all the information from the pointer and we end up with an invalid address.

-->

> 例2.2。類型不匹配時的警告示例 (Example of warnings when types are not matched)

```c
/*
 * types.c
 */

#include <stdio.h>
#include <stdint.h>

int main(void)
{
	char a;
	char *p = "hello";

	int i;

	// moving a larger variable into a smaller one
	i = 0x12341234;
	a = i;
	i = a;
	printf("i is %d\n", i);

	// moving a pointer into an integer
	printf("p is %p\n", p);
	i = p;
	// "fooling" with casts
	i = (int)p;
	p = (char*)i;
	printf("p is %p\n", p);

	return 0;
}
```

```
$ uname -m
i686

$ gcc -Wall -o types types.c
types.c: In function 'main':
types.c:19: warning: assignment makes integer from pointer without a cast

$ ./types
i is 52
p is 0x80484e8
p is 0x80484e8

$ uname -m
ia64

$ gcc -Wall  -o types types.c
types.c: In function 'main':
types.c:19: warning: assignment makes integer from pointer without a cast
types.c:21: warning: cast from pointer to integer of different size
types.c:22: warning: cast to pointer from integer of different size

$ ./types
i is 52
p is 0x40000000000009e0
p is 0x9e0

```

### 數字表示法 Number Representation

### 負值 Negative Values

在現代的10進制下，我們通過在它前面放一個負號來表示一個負數。當使用二進制時，我們需要使用不同的系統來表示負數。

在現代硬件上只有一種常用方案，但C99為負值表示定義了三種可接受的方法。

<!--
With our modern base 10 numeral system we indicate a negative number by placing a minus (-) sign in front of it. When using binary we need to use a different system to indicate negative numbers.

There is only one scheme in common use on modern hardware, but C99 defines three acceptable methods for negative value representation.

-->

### 符號位 Sign Bit

最直接的方法是簡單地說，數字的一個位表示一個負數或正數，這取決於它是否被設置。


這類似於擁有+和-的數學方法。這是相當合理的，一些原始計算機確實用這種方式表示負數。但是，使用二進制數字會帶來其他一些可能性，使硬件設計人員的工作更加輕鬆。


但是，注意0現在有兩個等價的值;一個有符號位集，一個沒有符號位集。有時這些值分別稱為+0和-0。

<!--
The most straight forward method is to simply say that one bit of the number indicates either a negative or positive value depending on it being set or not.

This is analogous to mathematical approach of having a + and -. This is fairly logical, and some of the original computers did represent negative numbers in this way. But using binary numbers opens up some other possibilities which make the life of hardware designers easier.

However, notice that the value 0 now has two equivalent values; one with the sign bit set and one without. Sometimes these values are referred to as +0 and -0 respectively.

-->

### 1 補數 One's Complement

補碼就是將not運算應用於正數來表示負數。例如，值-90 (-0x5A)用~01011010 = 10100101[4]表示


使用這種方案的最大優點是，在正數和負數之間添加一個負數並不需要特殊的邏輯，只需要將剩餘的任何進位加回最終值即可。考慮

<!--
One's complement simply applies the not operation to the positive number to represent the negative number. So, for example the value -90 (-0x5A) is represented by ~01011010 = 10100101[4]

With this scheme the biggest advantage is that to add a negative number to a positive number no special logic is required, except that any additional carry left over must be added back to the final value. Consider

-->

> 表2.15。一補數的加法 (One's Complement Addition)

十進制 Decimal | 二進制 Binary | 運算 Op
----|------------|---------
-90 | 	10100101 | 	+
100 | 	01100100 | 	 
--- | 	-------- | 	 
10 | 	100001001 | 	9
  | 	00001010 | 	10

如果你一個一個地加位，你會發現你最後會得到一個進位(上面高亮顯示)。通過將這個值添加到原始值，我們得到了正確的值10


我們仍然有兩個0被表示的問題。再一次，現代計算機沒有使用一補數，主要是因為有一個更好的方案。

<!--
If you add the bits one by one, you find you end up with a carry bit at the end (highlighted above). By adding this back to the original we end up with the correct value, 10

Again we still have the problem with two zeros being represented. Again no modern computer uses one's complement, mostly because there is a better scheme.

-->

### 2 補數 (Two's Complement)

2的補數和1的補數是一樣的，除了負的表示加了一個，然後我們就把剩下的進位去掉。繼續前面的例子，-90是~01011010+1=10100101+1 = 10100110。


這意味著在數字中有一個稍微奇怪的對稱可以被表示;例如一個8位整數2^8 = 256可能的值;用符號位表示，我們可以表示-127到127但用2補數，我們可以表示-127到128。這是因為我們去掉了兩個0的問題;假設“- 0”為(~00000000 +1)=(11111111+1)=00000000(注意丟棄進位)。

> 表 2.16 - 2 補數的加法 (Two's Complement Addition)

十進位 | 二進位 |	運算
--------|------|---------
-90 | 	10100110 | 	+
100 | 	01100100 | 	 
--- | 	-------- | 	 
10 | 	00001010	 |  

您可以看到，那些設計 2 補數硬件的人員，只需要為加法電路提供邏輯;減法可以通過將某值取 2 補數後加上新的值來實現。


類似地，你也可以用重複加法和重複減法除法來實現乘法。因此，2的補數可以將所有簡單的數學運算簡化為加法!


所有現代計算機都使用 2 補數表示法。

<!--
Two's complement is just like one's complement, except the negative representation has one added to it and we discard any left over carry bit. So to continue with the example from before, -90 would be ~01011010+1=10100101+1 = 10100110.

This means there is a slightly odd symmetry in the numbers that can be represented; for example with an 8 bit integer we have 2^8 = 256 possible values; with our sign bit representation we could represent -127 thru 127 but with two's complement we can represent -127 thru 128. This is because we have removed the problem of having two zeros; consider that "negative zero" is (~00000000 +1)=(11111111+1)=00000000 (note discarded carry bit).


You can see that by implementing two's complement hardware designers need only provide logic for addition circuits; subtraction can be done by two's complement negating the value to be subtracted and then adding the new value.

Similarly you could implement multiplication with repeated addition and division with repeated subtraction. Consequently two's complement can reduce all simple mathematical operations down to addition!

All modern computers use two's complement representation.

-->

### 符號擴展 Sign-extension

由於2補數格式，當增加有符號值的長度時，符號位元必須小心處理。


例如，32位int -10的值將在兩個補碼二進制文件中表示為111111111111111111110110。如果要將其轉換為64位的long long int，則需要確保將額外的32位設置為1，以保持原來的符號相同。


由於有了2補數，將現有值的頂部位替換為這個值就足夠了。這個過程被稱為符號擴展，通常由編譯器在語言標準定義的情況下處理，處理器通常提供特殊的指令來獲取一個值並將其擴展到某個更大的值。

<!--
Because of two's complement format, when increasing the size of signed value, it is important that the additional bits be sign-extended; that is, copied from the top-bit of the existing value.

For example, the value of an 32-bit int -10 would be represented in two's complement binary as 11111111111111111111111111110110. If one were to cast this to a 64-bit long long int, we would need to ensure that the additional 32-bits were set to 1 to maintain the same sign as the original.

Thanks to two's complement, it is sufficient to take the top bit of the exiting value and replace all the added bits with this value. This processes is referred to as sign-extension and is usually handled by the compiler in situations as defined by the language standard, with the processor generally providing special instructions to take a value and sign-extended it to some larger value.

-->

### 浮點數 Floating Point

到目前為止，我們只討論了整數; 接下來要討論浮點數。


為了創建一個小數，我們需要一些方法來表示二進制中的小數的概念。最常見的方案是IEEE-754浮點標準，因為該標準是由電氣和電子工程師協會發佈的。這個方案在概念上相當簡單，有點類似於“科學符號”。


在科學符號中，值123.45通常表示為1.2345x10<sup>2</sup>。1.2345 是尾數 (mantissa 或 significand)，10 是基底 (radix)，而 2 是指數 (exponent)。


在IEEE浮點模型中，我們分解可用位來表示十進制數的符號、尾數和指數。一個十進制數由 sign × significand × 2<sup>^exponent</sup>。


符號位等於1或-1。因為我們是二進制的，所以我們總是有隱含的基數2。


對於浮點值有不同的寬度——我們在下面只檢視 32 位元的浮點數。位數更多則會允許更大的精度。

<!--
So far we have only discussed integer or whole numbers; the class of numbers that can represent decimal values is called floating point.

To create a decimal number, we require some way to represent the concept of the decimal place in binary. The most common scheme for this is known as the IEEE-754 floating point standard because the standard is published by the Institute of Electric and Electronics Engineers. The scheme is conceptually quite simple and is somewhat analogous to "scientific notation".

In scientific notation the value 123.45 might commonly be represented as 1.2345x102. We call 1.2345 the mantissa or significand, 10 is the radix and 2 is the exponent.

In the IEEE floating point model, we break up the available bits to represent the sign, mantissa and exponent of a decimal number. A decimal number is represented by sign × significand × 2^exponent.

The sign bit equates to either 1 or -1. Since we are working in binary, we always have the implied radix of 2.

There are differing widths for a floating point value -- we examine below at only a 32 bit value. More bits allows greater precision.

-->

> 表2.17。IEEE浮點 (IEEE Floating Point)

符號 Sign | 指數 Exponent | 尾數 Significand/Mantissa
--|----------|--------------------------
S | EEEEEEEE | MMMMMMMMMMMMMMMMMMMMMMM

另一個重要的因素是指數的偏差。指數 Exponent 需要能夠同時表示正值和負值，因此從指數中減去隱含值127。例如，指數 0 代表 127, 指數 1 代表 128 ,而指數 126 代表 -1。


尾數的每個 bit 都會增加一點精度。考慮一下198765值的科學表示法表示。我們可以寫成 1.98765x10^6，對應於下面的表示

<!--
The other important factor is bias of the exponent. The exponent needs to be able to represent both positive and negative values, thus an implied value of 127 is subtracted from the exponent. For example, an exponent of 0 has an exponent field of 127, 128 would represent 1 and 126 would represent -1.

Each bit of the significand adds a little more precision to the values we can represent. Consider the scientific notation representation of the value 198765. We could write this as 1.98765x106, which corresponds to a representation below

-->

> 表2.18。科學記數法 1.98765x10^6 (Scientific Notation for 1.98765x10^6)

100 | 	. | 	10<sup>-1</sup> | 	10<sup>-2</sup> | 	10<sup>-3</sup> | 	10<sup>-4</sup> | 	10<sup>-5</sup>
--|-------|-------|-------|-------|-------|----------
1 | 	. | 	9 | 	8 | 	7 | 	6 | 	5

每一個額外的數字允許我們可以表示更大範圍的十進制值。在以10為基數的情況下，小數點後每一位數字的精度都要提高10倍。例如，我們可以用一位小數表示0.0到0.9(10個值)，用兩位數字表示0.00到0.99(100個值)，以此類推。在二進制中，不是每一個額外的數字給我們10倍的精度，我們只得到2倍的精度，如下表所示。這意味著二進製表示並不總是直接映射到十進製表示。

<!--
Each additional digit allows a greater range of decimal values we can represent. In base 10, each digit after the decimal place increases the precision of our number by 10 times. For example, we can represent 0.0 through 0.9 (10 values) with one digit of decimal place, 0.00 through 0.99 (100 values) with two digits, and so on. In binary, rather than each additional digit giving us 10 times the precision, we only get two times the precision, as illustrated in the table below. This means that our binary representation does not always map in a straight-forward manner to a decimal representation.

-->

> 表 2.19。二進制尾數 (Significands in binary)

20 | 	. | 2<sup>-1</sup> | 2<sup>-2</sup> | 2<sup>-3</sup> | 2<sup>-4</sup> | 2<sup>-5</sup>
--|-------|-------|-------|-------|-------|----------
1 | 	. | 	1/2 | 	1/4 | 	1/8 | 	1/16 | 	1/32
1 | 	. | 	0.5 | 	0.25 | 	0.125 | 0.0625 | 0.03125

由於只有一點精度，我們的分數精度不是很大;我們只能說分數要麼是0，要麼是0。5。如果我們再加上一點精度，現在我們可以說小數是0 0。25 0。5 0。75。通過另一種精度，我們現在可以表示值0、0.125、0.25、0.375、0.5、0.625、0.75、0.875。


因此，增加比特數可以使我們的精度越來越高。然而，由於可能的數字的範圍是無限的，我們永遠不會有足夠的位來表示任易精度的值。


例如，如果我們只有兩位精度並且需要表示值 0.3 我們只能說它最接近0.25; 顯然，這對於大多數應用程序來說都是不夠的。有 22 位的意義，我們有一個更好的分辨率，但它仍然是不夠的大多數應用。雙精度 double 將有效位的數量增加到 52 (它也增加了指數值的範圍)。有些硬件有一個 84 位浮點數，有整整64位的尾數。64位允許極高的精度，並且應該適用於除最苛刻的應用程序之外的所有應用程序 

<!-- (XXX是否足以表示長度小於一個原子的大小?) -->

<!--
With only one bit of precision, our fractional precision is not very big; we can only say that the fraction is either 0 or 0.5. If we add another bit of precision, we can now say that the decimal value is one of either 0,0.25,0.5,0.75. With another bit of precision we can now represent the values 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875.

Increasing the number of bits therefore allows us greater and greater precision. However, since the range of possible numbers is infinite we will never have enough bits to represent any possible value.

For example, if we only have two bits of precision and need to represent the value 0.3 we can only say that it is closest to 0.25; obviously this is insufficient for most any application. With 22 bits of significand we have a much finer resolution, but it is still not enough for most applications. A double value increases the number of significand bits to 52 (it also increases the range of exponent values too). Some hardware has an 84-bit float, with a full 64 bits of significand. 64 bits allows a tremendous precision and should be suitable for all but the most demanding of applications (XXX is this sufficient to represent a length to less than the size of an atom?)

-->

> 例 2.3。單精度與雙精度浮點數 (Floats versus Doubles)

```c
$ cat float.c
#include <stdio.h>

int main(void)
{
        float a = 0.45;
        float b = 8.0;

        double ad = 0.45;
        double bd = 8.0;

        printf("float+float, 6dp    : %f\n", a+b);
        printf("double+double, 6dp  : %f\n", ad+bd);
        printf("float+float, 20dp   : %10.20f\n", a+b);
        printf("dobule+double, 20dp : %10.20f\n", ad+bd);

        return 0;
}

$ gcc -o float float.c

$ ./float
float+float, 6dp    : 8.450000
double+double, 6dp  : 8.450000
float+float, 20dp   : 8.44999998807907104492
dobule+double, 20dp : 8.44999999999999928946

$ python
Python 2.4.4 (#2, Oct 20 2006, 00:23:25)
[GCC 4.1.2 20061015 (prerelease) (Debian 4.1.1-16.1)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> 8.0 + 0.45
8.4499999999999993
```

上面舉例說明了一個實際的例子。注意，對於printf給出的默認6位小數精度，兩個答案都是相同的，因為它們都是四捨五入的。然而，當要求以更大的精度給出結果時，在這種情況下是20位小數，我們可以看到結果開始出現分歧。使用double的代碼有更精確的結果，但仍然不是完全正確的。我們還可以看到，程序員若不小心處理浮點的話仍然會造成變數精度的問題!

<!--
A practical example is illustrated above. Notice that for the default 6 decimal places of precision given by printf both answers are the same, since they are rounded up correctly. However, when asked to give the results to a larger precision, in this case 20 decimal places, we can see the results start to diverge. The code using doubles has a more accurate result, but it is still not exactly correct. We can also see that programmers not explicitly dealing with float values still have problems with precision of variables!

-->

### 正規化 Normalised Values

在科學符號中，我們可以用許多不同的方式表示一個值。例如，10023x10<sup>0</sup> = 1002.3x10<sup>1</sup> = 100.23x10<sup>2</sup>。因此，我們將規範化的版本定義為1/radix <= significant < 1。在二進制中，這確保了最左邊的significant 位元總是1。

透過這一點，我們可以獲得額外的精度，在標準裏讓最左邊的永遠是1，因此可以不需要表達 (是隱含的)。

<!--
In scientific notation, we can represent a value in many different ways. For example, 10023x10<sup>0</sup> = 1002.3x10<sup>1</sup> = 100.23x10<sup>2</sup>. We thus define the normalised version as the one where 1/radix <= significand < 1. In binary this ensures that the leftmost bit of the significand is always one. Knowing this, we can gain an extra bit of precision by having the standard say that the leftmost bit being one is implied.

-->

> 表2.20。正規化0.375的例子 (Example of normalising 0.375)

2<sup>0</sup> | . | 2<sup>-1</sup> | 2<sup>-2</sup> |2<sup>-3</sup> |2<sup>-4</sup> |2<sup>-5</sup> |Exponent | 	Calculation
---|------|-------|-------|-------|-------|-------|---------|------------------------
0 | 	. | 	0 | 	1 | 	1 | 	0 | 	0 | 	2^0 | 	(0.25+0.125) × 1 = 0.375
0 | 	. | 	1 | 	1 | 	0 | 	0 | 	0 | 	2^-1 | 	(0.5+0.25)×.5=0.375
1 | 	. | 	1 | 	0 | 	0 | 	0 | 	0 | 	2^-2 | 	(1+0.5)×0.25=0.375

正如您在上面看到的，我們可以通過向上移動位使值標準化，只要我們通過增加指數來補償。

<!--
As you can see above, we can make the value normalised by moving the bits upwards as long as we compensate by increasing the exponent.

-->

### 正規化技巧 Normalisation Tricks

程序員面臨的一個常見問題是在位域中查找第一個集合位。考慮bitfield 0100;從右邊開始，第一個 1 出現在 bit 2 (從 0 開始，這是慣例)。


找到這個值的標準方法是向右移動，檢查最上面的位是否為1，然後終止或重複。這是一個緩慢的過程;如果位欄位是64位長，並且只設置了最後一個位，那麼您必須遍歷所有前面的63位!


然而，如果這個位的值是浮點數的符號，我們將它標準化，指數的值會告訴我們它移位了多少次。數字規範化的過程通常內置於處理器的浮點硬件單元中，因此操作非常快; 通常比重複移位和檢查操作快得多。


下面的示常式序演示了在Itanium處理器上查找第一個設置位的兩種方法。與大多數服務器處理器一樣，Itanium支持80位擴展浮點類型，具有64位的含義。這意味著一個未簽名的長符合長的雙重含義。當該值被加載時，它被規範化，因此通過讀取指數值(減去16位偏差)，我們可以看到它移動了多遠。

<!--
A common problem programmers face is finding the first set bit in a bitfield. Consider the bitfield 0100; from the right the first set bit would be bit 2 (starting from zero, as is conventional).

The standard way to find this value is to shift right, check if the uppermost bit is a 1 and either terminate or repeat. This is a slow process; if the bitfield is 64 bits long and only the very last bit is set, you must go through all the preceding 63 bits!

However, if this bitfield value were the signficand of a floating point number and we were to normalise it, the value of the exponent would tell us how many times it was shifted. The process of normalising a number is generally built into the floating point hardware unit on the processor, so operates very fast; usually much faster than the repeated shift and check operations.

The example program below illustrates two methods of finding the first set bit on an Itanium processor. The Itanium, like most server processors, has support for an 80-bit extended floating point type, with a 64-bit significand. This means a unsigned long neatly fits into the significand of a long double. When the value is loaded it is normalised, and and thus by reading the exponent value (minus the 16 bit bias) we can see how far it was shifted.

-->

> 例 2.4。程序找到第一個設置位 (Program to find first set bit)

```c
#include <stdio.h>

int main(void)
{
	//  in binary = 1000 0000 0000 0000
	//  bit num     5432 1098 7654 3210
	int i = 0x8000;
	int count = 0;
	while ( !(i & 0x1) ) {
		count ++;
		i = i >> 1;
	}
	printf("First non-zero (slow) is %d\n", count);

	// this value is normalised when it is loaded
	long double d = 0x8000UL;
	long exp;

	// Itanium "get floating point exponent" instruction
	asm ("getf.exp %0=%1" : "=r"(exp) : "f"(d));

	// note exponent include bias
	printf("The first non-zero (fast) is %d\n", exp - 65535);

}
```

### 把它放在一起 Bringing it together

在下面的示例代碼中，我們提取浮點數的組件並輸出它所表示的值。這只適用於IEEE格式的32位浮點值;然而，對於大多數具有浮點類型的體系結構來說，這是很常見的。

<!--
In the example code below we extract the components of a floating point number and print out the value it represents. This will only work for a 32 bit floating point value in the IEEE format; however this is common for most architectures with the float type.

-->

> 例2.5。檢查浮點 (Examining Floats)

```c
#include <stdio.h>

int main(void)
{
	//  in binary = 1000 0000 0000 0000
	//  bit num     5432 1098 7654 3210
	int i = 0x8000;
	int count = 0;
	while ( !(i & 0x1) ) {
		count ++;
		i = i >> 1;
	}
	printf("First non-zero (slow) is %d\n", count);

	// this value is normalised when it is loaded
	long double d = 0x8000UL;
	long exp;

	// Itanium "get floating point exponent" instruction
	asm ("getf.exp %0=%1" : "=r"(exp) : "f"(d));

	// note exponent include bias
	printf("The first non-zero (fast) is %d\n", exp - 65535);

}
```

我們之前檢查過的值8.45的示例輸出如下所示。

<!--
Sample output of the value 8.45, which we previously examined, is shown below.

-->

> 例 2.6。 分析8.45 (Analysis of 8.45)

```
$ ./float 8.45
8.450000 = 1 * (1/2^0 + 1/2^5 + 1/2^6 + 1/2^7 + 1/2^10 + 1/2^11 + 1/2^14 + 1/2^15 + 1/2^18 + 1/2^19 + 1/2^22 + 1/2^23) * 2^3
8.450000 = 1 * (1/1 + 1/32 + 1/64 + 1/128 + 1/1024 + 1/2048 + 1/16384 + 1/32768 + 1/262144 + 1/524288 + 1/4194304 + 1/8388608) * 2^3
8.450000 = 1 * 1.05624997616 * 8.000000
8.450000 = 8.44999980927
```

從這個例子中，我們瞭解了不精確是如何潛入到浮點數中的。

<!--
From this example, we get some idea of how the inaccuracies creep into our floating point numbers.

-->

> [1] 經常會看到參數的名稱被省略，並且只指定參數的類型。這允許實現者指定自己的參數名，以避免編譯器發出警告。
>
> [2] 一個雙下劃線函數__foo可以被稱為“dunder foo”。
> 
> [3] 注意，C99還有printf的可移植性助手。`<inttypes.h>` 可以用作指定大小類型的說明符。再次查看標準或拉開標題的完整信息。
> 
> [4] ~運算符是C語言運算符，用於不應用於該值。它有時也被稱為補碼運算符，原因顯而易見!

<!--
[1] Often you will see that the names of the parameters are omitted, and only the type of the parameter is specified. This allows the implementer to specify their own parameter names avoiding warnings from the compiler.

[2] A double-underscore function __foo may conversationally be referred to as "dunder foo".

[3] Note that C99 also has portability helpers for printf. The PRI macros in `<inttypes.h>` can be used as specifiers for types of specified sizes. Again see the standard or pull apart the headers for full information.

[4] The ~ operator is the C language operator to apply NOT to the value. It is also occasionally called the one's complement operator, for obvious reasons now!

-->
