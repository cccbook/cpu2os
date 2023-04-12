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

        printf("Producer: produce %d, buffer has %d data\n", data, count);

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

        printf("Consumer: consume %d, buffer has %d data\n", data, count);

        // 消費後發送 empty signal
        pthread_cond_signal(&empty);
        pthread_mutex_unlock(&mutex);

        // 模擬消費間隔
        sleep(rand() % 3);
    }

    pthread_exit(NULL);
}
