# kilo

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

## 

First we make sure it responded with an escape sequence. Then we pass a pointer to the third character of buf to sscanf(), skipping the '\x1b' and '[' characters. So we are passing a string of the form 24;80 to sscanf(). We are also passing it the string %d;%d which tells it to parse two integers separated by a ;, and put the values into the rows and cols variables.

Our fallback method for getting the window size is now complete. You should see that editorDrawRows() prints the correct number of tildes for the height of your terminal.
