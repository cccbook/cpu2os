ccc: 請解說下列程式碼

```c
#include <pthread.h>     // 引用 pthread 函式庫
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h> 

void *print_george(void *argu) {    // 每隔一秒鐘印出一次 George 的函數
  while (1) {    
    printf("George\n");    
    sleep(1);    
  }    
  return NULL;    
}    

void *print_mary(void *argu) {     // 每隔2秒鐘印出一次 Mary 的函數
  while (1) {    
    printf("Mary\n");    
    sleep(2);    
  }    
  return NULL;    
}    

int main() {     // 主程式開始
  pthread_t thread1, thread2;     // 宣告兩個執行緒
  pthread_create(&thread1, NULL, &print_george, NULL);    // 執行緒 print_george
  pthread_create(&thread2, NULL, &print_mary, NULL);    // 執行緒 print_mary
  while (1) {     // 主程式每隔一秒鐘
    printf("----------------\n");    // 就印出分隔行
    sleep(1);     // 停止一秒鐘
  }    
  return 0;    
}

```

ChatGPT:

這段程式碼使用 pthread 函式庫（POSIX Thread）來建立多執行緒的程式。主程式 main() 創建了兩個執行緒 thread1 和 thread2，它們執行分別執行 print_george() 和 print_mary() 這兩個函數。

print_george() 與 print_mary() 函數中都包含一個無窮迴圈，並且會在迴圈中每隔一段時間印出一個字串（分別是 "George" 和 "Mary"），這是通過 printf() 函數和 sleep() 函數實現的。

主程式 main() 中也包含一個無窮迴圈，每隔一秒鐘就會印出一個分隔行 "---------------"，然後停止一秒鐘。由於主程式與其他兩個執行緒是並行執行的，因此當主程式印出分隔行時，可能會在 "George" 或 "Mary" 字串的中間打斷。因此，這個程式可以展示出多執行緒的並行執行過程。

在 pthread_create() 函數中，第一個引數是指向執行緒 ID 的指標。 NULL 可以使用預設屬性創建執行緒。第二個引數是指向執行緒屬性的指標，如果使用預設屬性則可以設置為 NULL。第三個引數是指向函數的指標，這個函數是執行緒所要執行的函數。最後一個引數是可傳遞到該函數的參數。在這個程式中，由於 print_george() 和 print_mary() 函數不需要傳遞參數，因此都被設置為 NULL。

當兩個執行緒被創建後，它們會與主程式同時運行，執行各自的函數，實現並行運行。當程式運行時，你可以看到 "George" 和 "Mary" 字串的交替輸出，並且 "----------------" 字串被不定時地插入其中。

