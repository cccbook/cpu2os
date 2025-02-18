

叫 ChatGPT 寫個 B+Tree 是可以的

只是能不能正確跑還沒測

只要叫他先寫 C 的 header 

然後再一個函數一個函數貼給他就行了

* https://chat.openai.com/share/3fc66ecb-fa68-413e-bf73-36921c03ea83

## 結果，可以跑

```
ccckmit@asus MINGW64 /d/ccc/ccc112a/cpu2os/02-軟體/A1-資料庫/02b-chatBpTree (master)
$ ./a
B+Tree after insertions:
Level 0: 10
Level 1: 5 6 7
Level 1: 12 17 20 30

B+Tree after deletions:
Level 0: 10
Level 1: 5 7
Level 1: 12 17 20
```