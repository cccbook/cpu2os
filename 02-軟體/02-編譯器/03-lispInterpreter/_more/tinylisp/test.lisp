(+ 3 5)
(+ 2 7)
((lambda (x y) (+ x y)) 3 4)
(define add (lambda (x y) (+ x y))) (add 3 4)
(define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1))))))
fact
(fact 3)
(fact 10)
(if (> 6 5) (+ 1 1) (+ 2 2))
(if (< 6 5) (+ 1 1) (+ 2 2))
(define twice (lambda (x) (* 2 x)))
twice
(twice 5)
