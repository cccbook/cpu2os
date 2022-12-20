## 快取記憶體的影響力實驗

### 前言

話說在周星馳「功夫」這部電影裏，火雲邪神接住子彈之後說道：「天下武功，無堅不破，唯快不破！」 。

通常程式人之所以用 C 語言，主要原因有二，一是因為「快」、二是因為「指標在嵌入式系統上的用途」。

但是、同樣是用 C 語言，有些人的程式快如脫兔，而另一些人的程式卻慢如蝸牛，為何會有這樣的差異呢？

要能夠讓 C 語言快，必須瞭解「目標平台的計算機結構」，像是「管線、快取、記憶體管理、堆疊與堆積」等等，有時也要瞭解編譯器會如何編議你的程式。

在本文中，我們將利用一個「矩陣相乘」的範例，說明「快取」與「區域性」這兩個概念對程式速度的影響。

### 矩陣相乘速度評測

廢話不多說，讓我們直接來看這個「矩陣相乘」的測試程式，看完後再來分析為何會有很多倍的速度差異。

檔案：matrix.c

```CPP
#include <stdio.h>
#include <time.h>

#define N    1000
#define TYPE int

TYPE A[N][N], B[N][N], C[N][N];

void init() {
  int i,j,k;
  for (i=0; i<N; i++)
    for (j=0; j<N; j++) {
	  A[i][j] = i+j;
	  B[i][j] = i+j;
	  C[i][j] = 0;
	}
}

void mmul_ijk() {
  int i,j,k;
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
	  for (k=0; k<N; k++)
	    C[i][j] += A[i][k] * B[k][j];
}

void mmul_ikj() {
  int i,j,k;
  for (i=0; i<N; i++)
    for (k=0; k<N; k++)
      for (j=0; j<N; j++)
	    C[i][j] += A[i][k] * B[k][j];
}

void mmul_jki() {
  int i,j,k;
  for (j=0; j<N; j++)
    for (k=0; k<N; k++)
      for (i=0; i<N; i++)
	    C[i][j] += A[i][k] * B[k][j];
}

void run(void (*mmul)(), char *fname) {
  printf("========= %s ============\n", fname);
  time_t start, stop;
  init();
  start = time(NULL);
  printf("start=%d\n", start);
  mmul();
  stop  = time(NULL);
  printf("stop =%d\n", stop);
  printf("diff =%d\n", stop-start);
}

int main() {
  run(mmul_ijk, "mmul_ijk");
  run(mmul_ikj, "mmul_ikj");
  run(mmul_jki, "mmul_jki");
}
```

執行結果：

```
D:\c\cache>gcc -O0 matrix.c -o matrix

D:\c\cache>matrix
========= mmul_ijk ============
start=1388743938
stop =1388743953
diff =15
========= mmul_ikj ============
start=1388743953
stop =1388743958
diff =5
========= mmul_jki ============
start=1388743958
stop =1388743989
diff =31
```

您可以看到，mmul_ikj() 只花了 5 秒，比起 mmul_jki() 的 31 秒快上了六倍，究竟為何如此呢？

### 快取與區域性

在上述程式中，我們宣告了三組 `1000*1000` 的整數矩陣，每組大約耗用記憶體 1M 的整數大小，在筆者的電腦上，一個整數佔用 4byte，因此總共約耗用 `3*4MB=12MB` 的記憶體。

但是、筆者電腦的記憶體容量為 4G ，因此三個矩陣都可以完全放在記憶體中。

那麼、為何會有這麼大的速度差異呢？

根據「計算機結構的常識」推斷，原因應該在於快取記憶體，而要讓快取記憶體有效率的方式，在於增強程式的「區域性」 (locality)。

筆者用 dxdiag 指令檢視處理器，發現是 AMD Athlon II X4 645 Processor, 3.1 GHZ 的處理器，根據維基百科的資料，該處理器的快取大小如下：

```
L1 cache: 64 kB + 64 kB (data + instructions) per core
L2 cache: 1024 kB 
L3 cache: 無
```

因此 12MB 的資料無法完全放在 L2 cache 中，而 L1 cache 中所能放的資料量更少，所以在處理這個大小為 12MB 的矩陣相乘運算時，誰擁有最好的區域性，誰應該就會有最快的速度。

讓我們先看看以下這個最常見的標準矩陣相乘寫法 mmul_ijk()，您可以看到他的區域性並不會很好，因為最裏層的 B[k][j] 每次 k 都會變動一步，而 B 是 `1000*1000` 的矩陣，因此相當於每次 B 都要跳上 1000 個整數。 (不過 A[i][k] 的區域性還不錯，C[i][j] 的區域性則很好)。

```CPP
void mmul_ijk() {
  int i,j,k;
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
	  for (k=0; k<N; k++)
	    C[i][j] += A[i][k] * B[k][j];
}
```

所以、執行時上述的標準寫法花了 14 秒，在三者當中效能排行第二。


接著讓我們看看最快的 mmul_ikj()，該程式原始碼如下：

```CPP
void mmul_ikj() {
  int i,j,k;
  for (i=0; i<N; i++)
    for (k=0; k<N; k++)
      for (j=0; j<N; j++)
	    C[i][j] += A[i][k] * B[k][j];
}
``` 

由於最內層變動者為 j，因此 B[k][j] 有很好的區域性，而 A[i][k] 並不太變動，C[i][j] 也有很好的區域性，因此 mul_ikj() 程式的速度區域性最好，所以速度也最快。

最後讓我們來看效能最差的 mmul_jki()，由於 i 在最內層，但每次 A[i][k] 與 C[i][j] 的 i 變動時，都要跳上 1000 格的整數，因此其區域性是最糟的，所以效能也是最差的。

```CPP
void mmul_jki() {
  int i,j,k;
  for (j=0; j<N; j++)
    for (k=0; k<N; k++)
      for (i=0; i<N; i++)
	    C[i][j] += A[i][k] * B[k][j];
}
```

### 結語

雖然在此我們沒有詳細的討論快取記憶體的結構，但是光從區域性 (locality) 的角度來看，就能清楚的看出哪一個程式執行的最快，由於當今處理器的「快取與記憶體間的速度差異」越來越大 (一般來說可達 100 倍)，所以區域性越好的程式，通常速度也就會快上很多倍，這也是為何上述三個程式的表現差異如此之大的原因了。

因此、當您希望程式跑得快時，最好注意一下「區域性」結構是否良好，這很可能是決定程式效能的關鍵性因素。

### 參考文獻
* <http://en.wikipedia.org/wiki/Athlon_II>
* [深入理解计算机系统](http://www.waterlike.com.tw/bookdata.asp?NO=TP3C10C008)