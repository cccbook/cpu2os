ccc: 請問下列程式為何不會有 race condition?

```c
#include <stdio.h>
#include <pthread.h>

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
#define LOOPS 100000
int counter = 0;

void *inc()
{
  for (int i=0; i<LOOPS; i++) {
    pthread_mutex_lock( &mutex1 );
    counter = counter + 1;
    pthread_mutex_unlock( &mutex1 );
  }
  return NULL;
}

void *dec()
{
  for (int i=0; i<LOOPS; i++) {
    pthread_mutex_lock( &mutex1 );
    counter = counter - 1;
    pthread_mutex_unlock( &mutex1 );
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

ChatGPT:

這個程式使用了互斥量（mutex）來保證多個執行緒對 counter 變數的存取不會發生 race condition。

當一個執行緒執行到 pthread_mutex_lock 函數時，如果此時互斥量已經被其它執行緒佔用，則該執行緒會進入等待狀態，直到互斥量被釋放為止。當一個執行緒執行完對 counter 變數的操作之後，會執行 pthread_mutex_unlock 函數，釋放互斥量，讓其它執行緒可以繼續對 counter 變數進行操作。

由於互斥量同一時間只允許一個執行緒進入臨界區，因此可以保證多個執行緒對 counter 變數的存取不會發生 race condition。這種使用互斥量的方式稱為「鎖」（lock），因為當一個執行緒進入臨界區時，會將互斥量鎖定，使其它執行緒無法進入臨界區，等到該執行緒離開臨界區時，才會釋放互斥量，讓其它執行緒進入臨界區。
