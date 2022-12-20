#include <stdio.h>
#include <threads.h>
#include <stdatomic.h>

// pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
#define LOOPS 100000
atomic_int counter = 0;

int inc(void* data)
{
  (void)data;
  for (int i=0; i<LOOPS; i++) {
    // pthread_mutex_lock( &mutex1 );
    counter += 1; // 只能用 counter+=1 或 counter++， 不能用 counter = counter + 1 否則會有 race condition.
    // pthread_mutex_unlock( &mutex1 );
  }
  return 0;
}

int dec(void* data)
{
  (void)data;
  for (int i=0; i<LOOPS; i++) {
    // pthread_mutex_lock( &mutex1 );
    counter--; // counter = counter - 1;
    // pthread_mutex_unlock( &mutex1 );
  }
  return 0;
}


int main() 
{
	// pthread_t thread1, thread2;
  thrd_t thread1, thread2;

	thrd_create(&thread1, inc, NULL);
  thrd_create(&thread2, dec, NULL);

  thrd_join(thread1, NULL);
  thrd_join(thread2, NULL);
  printf("counter=%d\n", counter);
}