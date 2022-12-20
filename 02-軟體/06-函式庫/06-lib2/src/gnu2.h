#pragma once

// 以下的 _lambda 語法只能在 gcc 當中用
// https://stackoverflow.com/questions/10405436/anonymous-functions-using-gcc-statement-expressions
// https://github.com/wd5gnr/clambda/blob/master/clambda2.c
// https://hackaday.com/2019/09/11/lambdas-for-c-sort-of/
// https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html#Statement-Exprs
#define _lambda(lambda$_ret, lambda$_args, lambda$_body)\
  ({\
    lambda$_ret lambda$__anon$ lambda$_args\
      lambda$_body\
    &lambda$__anon$;\
  })

#define fx(type, exp) _lambda(type, (type x), { return exp; })
#define fxy(type, exp) _lambda(type, (type x, type y), { return exp; })

