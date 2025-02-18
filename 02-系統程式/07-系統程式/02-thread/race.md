請說明下列程式，為何會有 race condition ?

```c
#include <stdio.h>
#include <pthread.h>

#define LOOPS 100000000
int counter = 0;

void *inc()
{
  for (int i=0; i<LOOPS; i++) {
    counter = counter + 1;
  }
  return NULL;
}

void *dec()
{
  for (int i=0; i<LOOPS; i++) {
    counter = counter - 1;
  }
  return NULL;
}


int main() 
{
  pthread_t thread1, thread2;

  pthread_create(&thread1, NULL, inc, NULL);
  pthread_create(&thread2, NULL, dec, NULL);

  pthread_join( thread1, NULL);
  pthread_join( thread2, NULL);
  printf("counter=%d\n", counter);
}

```

上述程式中，兩個線程分別執行 inc() 和 dec() 函式，對 counter 變量進行加 1 和減 1 的操作。由於 counter 變量是全局變量，多個線程同時對其進行操作，因此會產生 race condition。

在上述程式中，race condition 發生的原因是，多個線程同時對 counter 變量進行寫入操作。由於這是一個非原子操作，涉及到多個步驟（讀取、計算、寫入），因此在執行過程中可能會發生不同步的情況。例如，在一個線程尚未完成整個寫入操作的時候，另一個線程已經對其進行了讀取，這樣就可能導致計算結果出現錯誤。

為了避免 race condition，可以使用同步機制（如 mutex、semaphore 等）來確保在同一時間只有一個線程能夠對 counter 變量進行寫入操作。例如，在 inc() 和 dec() 函式中都加入 mutex，確保在執行寫入操作時只有一個線程能夠進入臨界區，就可以避免 race condition 的問題。

## race.s 

從組合語言的觀點，counter ++ 會變成下列程式

```asm
.L3:
	movl	counter(%rip), %eax
	addl	$1, %eax
	movl	%eax, counter(%rip)
	addl	$1, -4(%rbp)
```

而 counter -- 則會變成下列程式

```asm
.L7:
	movl	counter(%rip), %eax
	subl	$1, %eax
	movl	%eax, counter(%rip)
	addl	$1, -4(%rbp)
```

這些程式以不同方式重疊，結果也會不同。

