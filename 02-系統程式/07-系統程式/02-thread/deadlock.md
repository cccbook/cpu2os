## ccc: 請問 deadlock  是甚麼?

ChatGPT:

Deadlock 是指在多執行緒或多進程系統中，當多個執行緒或進程互相等待對方釋放資源時，導致所有執行緒或進程都無法繼續執行的一種狀態。

當發生 deadlock 時，每個執行緒或進程都被阻塞，等待其它執行緒或進程釋放資源。但是，由於每個執行緒或進程都在等待對方釋放資源，因此系統無法進行任何進一步的處理，也無法解除 deadlock 狀態，進而導致整個系統癱瘓。

造成 deadlock 的原因通常是多個執行緒或進程在競爭同一組資源時，沒有按照相同的順序進行存取，導致某些執行緒或進程在等待其它執行緒或進程釋放資源時，自身卻無法釋放已經佔用的資源，從而導致死結。

要避免 deadlock，通常需要對系統進行合理的設計和管理，例如使用適當的同步機制、避免執行緒或進程無限期地等待資源等。此外，也可以使用一些死結檢測和解除死結的工具來解決 deadlock 的問題。

## ccc: 請問下列程式為何會有死結?

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

pthread_mutex_t x;
pthread_mutex_t y;

void *A(); 
void *B(); 

int main(int argc, char *argv[])
{
    pthread_t threadA, threadB;
    pthread_attr_t attr;

    pthread_attr_init(&attr);
    pthread_mutex_init(&x, NULL);
    pthread_mutex_init(&y, NULL);

    pthread_create(&threadA, &attr, A, NULL);
    pthread_create(&threadB, &attr, B, NULL);

    pthread_join(threadA, NULL);
    pthread_join(threadB, NULL);

    pthread_mutex_destroy(&x);
    pthread_mutex_destroy(&y);
}

void *A() 
{
    pthread_mutex_lock(&x);
    printf("A lock x\n");

    sleep(1);
    pthread_mutex_lock(&y);
    printf("A lock y\n");

    pthread_mutex_unlock(&y); 
    pthread_mutex_unlock(&x); 

    printf("finished A\n");

    pthread_exit(0);
}

void *B()
{
    
    pthread_mutex_lock(&y);
    printf("B lock y\n");
    sleep(1);
    pthread_mutex_lock(&x);
    printf("B lock x\n");
    pthread_mutex_unlock(&x);
    pthread_mutex_unlock(&y);

    pthread_exit(0);
}

```

ChatGPT:

此程式會有死結的原因為以下：

當執行緒 A 被啟動時，它鎖定了 mutex x，然後暫停了一秒。同時，當執行緒 B 被啟動時，它鎖定了 mutex y，然後暫停了一秒。接下來，執行緒 A 嘗試鎖定 mutex y，但是因為它已經被執行緒 B 鎖定了，所以執行緒 A 會等待 mutex y 被解鎖。同時，執行緒 B 也會嘗試鎖定 mutex x，但是它已經被執行緒 A 鎖定了，所以執行緒 B 也會等待 mutex x 被解鎖。這樣，兩個執行緒都等待著對方解鎖它們需要的 mutex，導致了死結。因此，此程式會有死結。要避免此問題，我們需要確保在使用多個 mutex 時，所有執行緒都按照相同的順序鎖定這些 mutex。
