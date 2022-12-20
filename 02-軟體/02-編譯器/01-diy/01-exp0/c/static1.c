// 產生下一個臨時變數的代號， ex: 3 代表 t3。
int nextTemp() {
  static int tempIdx = 0;
  return tempIdx++;
}