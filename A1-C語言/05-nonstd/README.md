# 非標準語法


## gnu only

* https://www.gnu.org/software/gnu-c-manual/gnu-c-manual.html
    * https://www.gnu.org/software/gnu-c-manual/gnu-c-manual.html#The-goto-Statement

```cpp

enum Play { ROCK=0, PAPER=1, SCISSORS=2 };
enum Result { WIN, LOSE, DRAW };

static enum Result turn (void) 
{
  const void * const jumptable[] = {&&rock, &&paper, &&scissors};
  enum Play opp;                /* opponent’s play */
  goto *jumptable[select_option (&opp)];
 rock:
  return opp == ROCK ? DRAW : (opp == PAPER ? LOSE : WIN);
 paper:
  return opp == ROCK ? WIN  : (opp == PAPER ? DRAW : LOSE);
 scissors:
  return opp == ROCK ? LOSE : (opp == PAPER ? WIN  : DRAW);
}

```
