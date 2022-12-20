# kilo

```
mac020:04-repos mac020$ 
```

## 解說

```cpp
void editorRefreshScreen() {
  write(STDOUT_FILENO, "\x1b[2J", 4);
  write(STDOUT_FILENO, "\x1b[H", 3);
}
```

This escape sequence is only 3 bytes long, and uses the H command (Cursor Position) to position the cursor. The H command actually takes two arguments: the row number and the column number at which to position the cursor. So if you have an 80×24 size terminal and you want the cursor in the center of the screen, you could use the command `<esc>`[12;40H. (Multiple arguments are separated by a ; character.) The default arguments for H both happen to be 1, so we can leave both arguments out and it will position the cursor at the first row and first column, as if we had sent the `<esc>`[1;1H command. (Rows and columns are numbered starting at 1, not 0.)