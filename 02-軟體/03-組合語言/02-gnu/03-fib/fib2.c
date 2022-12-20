int fib(int n) {
  int x=1,y=2,z=n+x+y;
  if (n <= 1) return 1;
    return fib(n-1) + fib(n-2);
}
