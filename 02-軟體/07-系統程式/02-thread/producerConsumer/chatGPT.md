# ccc:請寫一個生產者消費者的程式，先寫生產者

以下是一個簡單的生產者消費者程式，其中包含一個生產者和一個消費者。程式使用了互斥鎖來保護共享的緩衝區，以避免資料競爭。在此程式中，生產者不斷產生資料，並把資料放入緩衝區中，而消費者不斷從緩衝區中取出資料。

```c
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFFER_SIZE 10
#define MAX_ITEM 20

int buffer[BUFFER_SIZE];
int count = 0;
int in = 0;
int out = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t full = PTHREAD_COND_INITIALIZER;
pthread_cond_t empty = PTHREAD_COND_INITIALIZER;

void *producer(void *arg)
{
    int item;
    for (int i = 0; i < MAX_ITEM; i++)
    {
        item = rand() % 1000;  // 隨機產生一個資料
        pthread_mutex_lock(&mutex);
        while (count == BUFFER_SIZE)  // 緩衝區已滿，等待
        {
            pthread_cond_wait(&empty, &mutex);
        }
        buffer[in] = item;  // 將資料放入緩衝區
        in = (in + 1) % BUFFER_SIZE;
        count++;
        printf("Producer produces item %d\n", item);
        pthread_mutex_unlock(&mutex);
        pthread_cond_signal(&full);  // 喚醒等待中的消費者
        sleep(rand() % 5);  // 隨機等待一段時間
    }
    pthread_exit(NULL);
}

int main()
{
    pthread_t tid;
    srand(time(NULL));
    pthread_create(&tid, NULL, producer, NULL);
    pthread_join(tid, NULL);
    return 0;
}

```

在此程式中，我們首先宣告了一些常數，包括緩衝區的大小和生產的最大數量。然後我們宣告了一些全域變數，包括緩衝區本身以及一些與緩衝區相關的變數。接著我們宣告了互斥鎖和兩個條件變數。

生產者的主要任務是產生資料，將資料放入緩衝區中。在此程式中，我們使用了一個 for 迴圈來產生一定數量的資料。在迴圈內部，我們先隨機產生一個資料，然後使用互斥鎖來保護共享變數。如果緩衝區已滿，生產者會等待直到緩衝區有空

## ccc: 繼續

ChatGPT:

好的，接下來我們來寫消費者的程式碼：

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define BUFFER_SIZE 10

int buffer[BUFFER_SIZE];
int count = 0;
int in = 0;
int out = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t empty = PTHREAD_COND_INITIALIZER;
pthread_cond_t full = PTHREAD_COND_INITIALIZER;

void *producer(void *arg);
void *consumer(void *arg);

int main() {
    pthread_t producer_thread, consumer_thread;

    // 建立 producer 和 consumer thread
    pthread_create(&producer_thread, NULL, producer, NULL);
    pthread_create(&consumer_thread, NULL, consumer, NULL);

    // 等待 thread 結束
    pthread_join(producer_thread, NULL);
    pthread_join(consumer_thread, NULL);

    // 釋放 mutex 和 condition variable 的資源
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&empty);
    pthread_cond_destroy(&full);

    return 0;
}

void *producer(void *arg) {
    int data;

    for (int i = 0; i < 20; i++) {
        data = rand() % 100; // 產生一個 0-99 的隨機數

        pthread_mutex_lock(&mutex);
        // 當 buffer 已滿時，等待消費者消費後再繼續生產
        while (count == BUFFER_SIZE) {
            pthread_cond_wait(&empty, &mutex);
        }

        buffer[in] = data;
        in = (in + 1) % BUFFER_SIZE;
        count++;

        printf("Producer: 生產 %d，目前 buffer 有 %d 個資料\n", data, count);

        // 生產後發送 full signal
        pthread_cond_signal(&full);
        pthread_mutex_unlock(&mutex);

        // 模擬生產間隔
        sleep(rand() % 3);
    }

    pthread_exit(NULL);
}

void *consumer(void *arg) {
    int data;

    for (int i = 0; i < 20; i++) {
        pthread_mutex_lock(&mutex);
        // 當 buffer 為空時，等待生產者生產後再進行消費
        while (count == 0) {
            pthread_cond_wait(&full, &mutex);
        }

        data = buffer[out];
        out = (out + 1) % BUFFER_SIZE;
        count--;

        printf("Consumer: 消費 %d，目前 buffer 有 %d 個資料\n", data, count);

        // 消費後發送 empty signal
        pthread_cond_signal(&empty);
        pthread_mutex_unlock(&mutex);

        // 模擬消費間隔
        sleep(rand() % 3);
    }

    pthread_exit(NULL);
}

```

上述程式碼中，我們在 producer() 和 consumer() 中使用了 pthread_cond_wait() 和 pthread_cond_signal()，來實現 condition variable。當 buffer 已滿時，生產者會等待消費者消費後再繼續生產，此時生產者會呼叫 pthread_cond_wait()，