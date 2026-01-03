# gcc -sanitize=address

當你 gcc 加入 -fsanitize=address 時，有錯誤的話，會印出堆疊與行號。（有助於 debug）


```
gcc -Wall -Wextra -g -fsanitize=address -fsanitize=undefined jack2vm.c -o jack2vm
```