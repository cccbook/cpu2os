#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

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
