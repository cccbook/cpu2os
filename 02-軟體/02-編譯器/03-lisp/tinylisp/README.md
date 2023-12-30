# tinylisp

https://github.com/Robert-van-Engelen/tinylisp/blob/main/src/tinylisp.c

```
$ gcc tinylisp.c -o tinylisp

$ ./tinylisp
tinylisp
930>((lambda (x y) (+ x y)) 3 4)
7
929>(define add (lambda (x y) (+ x y))) (add 3 4)
add
899>7
899>(define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1))))))
fact
845>(fact 3)
1
845>(if (> 6 5) (+ 1 1) (+ 2 2))
2
845>(if (< 6 5) (+ 1 1) (+ 2 2))
4
845>(begin (define x 1) (set! x (+ x 1)) (+ x 1))
ERR
844>(define twice (lambda (x) (* 2 x)))
twice
815>(twice 5)
10
815>(define compose (lambda (f g) (lambda (x) (f (g x)))))
compose
773>((compose list twice) 5)
ERR
```