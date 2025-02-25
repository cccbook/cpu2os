# 系統程式簡介

* [nand2tetris -- Assembler](http://www.nand2tetris.org/lectures/PDF/lecture%2006%20assembler.pdf)

## 語法理論

* [人工智慧 -- 語言處理](https://mdbookspace.com/view/ai/linguistics.md)
* [計算語言學](https://mdbookspace.com/view/ccc/NLP.md)

```
// === EBNF Grammar =====
// E=T ([+-] T)*
// T=F ([*/] F)?
// T=[0-9] | (E)
```

## 英語生成

```
user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/03-compiler/00-gen (master)
$ gcc genEnglish.c rlib.c -o genEnglish

user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/03-compiler/00-gen (master)
$ ./genEnglish
the dog chase the cat

user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/03-compiler/00-gen (master)
$ ./genEnglish
a dog chase a cat

user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/03-compiler/00-gen (master)
$ ./genEnglish
the cat chase the dog

user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/03-compiler/00-gen (master)
$ ./genEnglish
the cat chase the dog
```

### 運算式生成

```

user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/03-compiler/00-gen (master)
$ gcc genExp.c rlib.c -o genExp

user@DESKTOP-96FRN6B MINGW64 /d/pmedia/陳鍾誠/課程/系統程式/03-compiler/00-gen (master)
$ ./genExp
7*(8+5/1)-1*2-8/1(7)/2
4/8
2
(1)/2
0*(5/0)-((4*6)*9)/((9/9)*6)
5*9
7*1
7*2+4*1
7
5

```

