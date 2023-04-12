#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define N 5
#define LEFT (i + N - 1) % N
#define RIGHT (i + 1) % N
#define THINKING 0
#define HUNGRY 1
#define EATING 2

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond[N]; // = PTHREAD_COND_INITIALIZER;

int state[N];
int phil[N] = {0, 1, 2, 3, 4};

void test(int i);

void philosopher(int i) {
    while(1) {
        printf("Philosopher %d is thinking.\n", i);
        sleep(rand() % 5 + 1); // think for a random time

        // acquire chopsticks
        pthread_mutex_lock(&mutex);
        state[i] = HUNGRY;
        printf("Philosopher %d is hungry.\n", i);
        test(i);
        while (state[i] != EATING) {
            pthread_cond_wait(&cond[i], &mutex);
        }
        pthread_mutex_unlock(&mutex);

        // eat for a random time
        printf("Philosopher %d is eating.\n", i);
        sleep(rand() % 5 + 1);

        // release chopsticks
        pthread_mutex_lock(&mutex);
        state[i] = THINKING;
        printf("Philosopher %d is done eating and now thinking.\n", i);
        test(LEFT);
        test(RIGHT);
        pthread_mutex_unlock(&mutex);
    }
}

void test(int i) {
    if (state[i] == HUNGRY && state[LEFT] != EATING && state[RIGHT] != EATING) {
        state[i] = EATING;
        pthread_cond_signal(&cond[i]);
    }
}

int main() {
    pthread_t thread_id[N];
    int i;
    srand(time(NULL));
    for (i=0; i<N; i++) {
        cond[i] = PTHREAD_COND_INITIALIZER;
    }
    for (i = 0; i < N; i++) {
        pthread_create(&thread_id[i], NULL, (void *) philosopher, (void *)&phil[i]);
    }
    for (i = 0; i < N; i++) {
        pthread_join(thread_id[i], NULL);
    }
    return 0;
}
