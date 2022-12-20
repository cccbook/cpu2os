# kilo

不能在 vs-code 使用，按 Ctrl-Q 可以離開

```























mac020:03-clrscr mac020$ 

```

## 原理

```cpp
void editorRefreshScreen() {
  write(STDOUT_FILENO, "\x1b[2J", 4);
}
```

The 4 in our write() call means we are writing 4 bytes out to the terminal. The first byte is \x1b, which is the escape character, or 27 in decimal. (Try and remember \x1b, we will be using it a lot.) The other three bytes are [2J.

We are writing an escape sequence to the terminal. Escape sequences always start with an escape character (27) followed by a [ character. Escape sequences instruct the terminal to do various text formatting tasks, such as coloring text, moving the cursor around, and clearing parts of the screen.

We are using the J command (Erase In Display) to clear the screen. Escape sequence commands take arguments, which come before the command. In this case the argument is 2, which says to clear the entire screen. `<esc>`[1J would clear the screen up to where the cursor is, and `<esc>`[0J would clear the screen from the cursor up to the end of the screen. Also, 0 is the default argument for J, so just `<esc>`[J by itself would also clear the screen from the cursor to the end.

