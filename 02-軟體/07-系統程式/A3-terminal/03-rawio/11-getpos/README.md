# kilo

## compile

```
mac020:11-getpos mac020$ make
cc kilo.c -o kilo -Wall -Wextra -pedantic -std=c99
kilo.c:64:28: warning: unused parameter 'rows' [-Wunused-parameter]
int getCursorPosition(int *rows, int *cols) {
                           ^
kilo.c:64:39: warning: unused parameter 'cols' [-Wunused-parameter]
int getCursorPosition(int *rows, int *cols) {
                                      ^
2 warnings generated.

```

## 

```cpp
int getCursorPosition(int *rows, int *cols) {
  char buf[32];
  unsigned int i = 0;
  if (write(STDOUT_FILENO, "\x1b[6n", 4) != 4) return -1;
  while (i < sizeof(buf) - 1) {
    if (read(STDIN_FILENO, &buf[i], 1) != 1) break;
    if (buf[i] == 'R') break;
    i++;
  }
  buf[i] = '\0';
  printf("\r\n&buf[1]: '%s'\r\n", &buf[1]);
  editorReadKey();
  return -1;
}
```
