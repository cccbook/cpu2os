# kilo

數字會變紅色

```
d
adfjka
adfl
dfal
02m.as03
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
[No Name] - 5 lines (modified)                                               5/5


```


##

```cpp
void editorDrawRows(struct abuf *ab) {
  int y;
  for (y = 0; y < E.screenrows; y++) {
    int filerow = y + E.rowoff;
    if (filerow >= E.numrows) {
      if (E.numrows == 0 && y == E.screenrows / 3) {
        char welcome[80];
        int welcomelen = snprintf(welcome, sizeof(welcome),
          "Kilo editor -- version %s", KILO_VERSION);
        if (welcomelen > E.screencols) welcomelen = E.screencols;
        int padding = (E.screencols - welcomelen) / 2;
        if (padding) {
          abAppend(ab, "~", 1);
          padding--;
        }
        while (padding--) abAppend(ab, " ", 1);
        abAppend(ab, welcome, welcomelen);
      } else {
        abAppend(ab, "~", 1);
      }
    } else {
      int len = E.row[filerow].rsize - E.coloff;
      if (len < 0) len = 0;
      if (len > E.screencols) len = E.screencols;
      char *c = &E.row[filerow].render[E.coloff];
      int j;
      for (j = 0; j < len; j++) {
        if (isdigit(c[j])) {
          abAppend(ab, "\x1b[31m", 5);
          abAppend(ab, &c[j], 1);
          abAppend(ab, "\x1b[39m", 5);
        } else {
          abAppend(ab, &c[j], 1);
        }
      }
    }
    abAppend(ab, "\x1b[K", 3);
    abAppend(ab, "\r\n", 2);
  }
}

```