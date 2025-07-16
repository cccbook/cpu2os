#include <lib2.h>

int square(int x) { return x*x; }
int cadd(int t, int x) { return t+x; }

void test1() {
  int a[]={ 1, 2, 3 };
  int n = LEN(a);
  OK(ARRAY_EQ(a, a, int, n));
 
  int b[n];
  // MAP(a, n, square, b);
  #define sq(x) x*x
  MAP(a, n, sq, b);
  EACH(b, n, PUT); BR;
  OK(!ARRAY_EQ(a, b, int, n));

  int r = 0;
  // #define cadd(t, x) (t+x)
  REDUCE(a, n, cadd, r);
  printf("r=%d\n", r);

  int m;
  #define odd(x) (x%2==1)
  FILTER(a, n, odd, b, m);
  printf("filter(a, odd)="); EACH(b, m, PUT); BR;

  #define even(x) (x%2==0)
  int eveni = FIND(a, n, even);
  OK(eveni==1);

  char str[100];
  STR(a, n, str); printf("a:str=%s\n", str);
  STR(b, n, str); printf("b:str=%s\n", str);
}

void test2() {
  OK(EQ(1, 1));
  OK(NE(1, 2));
  OK(LT(1.3, 2.5));
  OK(AND(true, true));
  OK(OR(1, 0));
  OK(NOT(false));
  OK(NEG(10.0)<1.3);
  OK(SHL(1, 3)==8);
  OK(MUL(3, 7)==21);
}

void test3() {
  int a[]={ 1, 2, 3 }, b[] = { 2, 3, 4 }, c[] = { 2, 2, 2 };
  int n = LEN(a);
  OK(n==3);
  OK(ALL(a, b, n, LT));
  OK(ANY(a, b, n, LT));
  OK(ANY(a, c, n, GT));
}

int main(void) {
  test1();
  test2();
  test3();
}
