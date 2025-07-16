/* tinylisp.c with NaN boxing by Robert A. van Engelen 2022 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define uint unsigned
#define f64 double
#define u64 unsigned long long
#define Tk(x) *(u64 *)&x >> 48
#define N 1024

uint hp = 0; // heap pointer
uint sp = N; // stack pointer
uint ATOM = 0x7ff8, PRIM = 0x7ff9, CONS = 0x7ffa, CLOS = 0x7ffb, NIL = 0x7ffc;
f64 cell[N], nil, tru, err, env;

f64 box(uint t, uint i)
{
    f64 x;
    *(u64 *)&x = (u64)t << 48 | i;
    return x;
}

uint ord(f64 x) { return *(u64 *)&x; }

f64 num(f64 n) { return n; }

uint equ(f64 x, f64 y) { return *(u64 *)&x == *(u64 *)&y; }

f64 atom(const char *s)
{
    uint i = 0;
    while (i < hp && strcmp((char *)cell + i, s))
        i += strlen((char *)cell + i) + 1;
    if (i == hp && (hp += strlen(strcpy((char *)cell + i, s)) + 1) > sp << 3)
        abort();
    return box(ATOM, i);
}

f64 cons(f64 x, f64 y)
{
    cell[--sp] = x;
    cell[--sp] = y;
    if (hp > sp << 3)
        abort();
    return box(CONS, sp);
}

f64 car(f64 p) { 
    return (Tk(p) & ~(CONS ^ CLOS)) == CONS ? cell[ord(p) + 1] : err; 
}

f64 cdr(f64 p) {
    return (Tk(p) & ~(CONS ^ CLOS)) == CONS ? cell[ord(p)] : err; 
}

f64 pair(f64 v, f64 x, f64 e) { return cons(cons(v, x), e); }

f64 closure(f64 v, f64 x, f64 e) { 
    return box(CLOS, ord(pair(v, x, equ(e, env) ? nil : e))); 
}

f64 assoc(f64 v, f64 e)
{
    while (Tk(e) == CONS && !equ(v, car(car(e))))
        e = cdr(e);
    return Tk(e) == CONS ? cdr(car(e)) : err;
}

uint not(f64 x) { return Tk(x) == NIL; }

uint let(f64 x) { return Tk(x) != NIL && !not(cdr(x)); }

f64 eval(f64, f64), parse();

f64 evlis(f64 t, f64 e) { 
    return Tk(t) == CONS ? cons(eval(car(t),e),evlis(cdr(t),e)) : 
           Tk(t) == ATOM ? assoc(t,e) : nil; 
}

f64 f_eval(f64 t, f64 e) { return eval(car(evlis(t, e)), e); }
f64 f_quote(f64 t, f64 _) { return car(t); }
f64 f_cons(f64 t, f64 e) { 
    return t = evlis(t, e), cons(car(t), car(cdr(t))); 
}
f64 f_car(f64 t, f64 e) { return car(car(evlis(t, e))); }
f64 f_cdr(f64 t, f64 e) { return cdr(car(evlis(t, e))); }
f64 f_add(f64 t, f64 e)
{
    f64 n = car(t = evlis(t, e));
    while (!not(t = cdr(t)))
        n += car(t);
    return num(n);
}
f64 f_sub(f64 t, f64 e)
{
    f64 n = car(t = evlis(t, e));
    while (!not(t = cdr(t)))
        n -= car(t);
    return num(n);
}
f64 f_mul(f64 t, f64 e)
{
    f64 n = car(t = evlis(t, e));
    while (!not(t = cdr(t)))
        n *= car(t);
    return num(n);
}
f64 f_div(f64 t, f64 e)
{
    f64 n = car(t = evlis(t, e));
    while (!not(t = cdr(t)))
        n /= car(t);
    return num(n);
}

f64 f_int(f64 t, f64 e)
{
    f64 n = car(evlis(t, e));
    return n < 1e16 && n > -1e16 ? (long long)n : n;
}

f64 f_lt(f64 t, f64 e) { 
    return t = evlis(t, e), car(t) - car(cdr(t)) < 0 ? tru : nil; 
}

f64 f_eq(f64 t, f64 e) { 
    return t = evlis(t, e), equ(car(t), car(cdr(t))) ? tru : nil; 
}

f64 f_not(f64 t, f64 e) { 
    return not(car(evlis(t, e))) ? tru : nil; 
}

f64 f_or(f64 t, f64 e)
{
    f64 x = nil;
    while (Tk(t) != NIL && not(x = eval(car(t), e)))
        t = cdr(t);
    return x;
}

f64 f_and(f64 t, f64 e)
{
    f64 x = nil;
    while (Tk(t) != NIL && !not(x = eval(car(t), e)))
        t = cdr(t);
    return x;
}

f64 f_cond(f64 t, f64 e)
{
    while (Tk(t) != NIL && not(eval(car(car(t)), e)))
        t = cdr(t);
    return eval(car(cdr(car(t))), e);
}

f64 f_if(f64 t, f64 e) { return eval(car(cdr(not(eval(car(t), e)) ? cdr(t) : t)), e); }

f64 f_leta(f64 t, f64 e)
{
    for (; let(t); t = cdr(t))
        e = pair(car(car(t)), eval(car(cdr(car(t))), e), e);
    return eval(car(t), e);
}

f64 f_lambda(f64 t, f64 e) { return closure(car(t), car(cdr(t)), e); }

f64 f_define(f64 t, f64 e)
{
    env = pair(car(t), eval(car(cdr(t)), e), env);
    return car(t);
}

// prim: primitive (基本函數)
struct
{
    const char *s;
    f64 (*f)(f64, f64);
} prim[] = {
    {"eval", f_eval}, {"quote", f_quote}, {"cons", f_cons}, 
    {"car", f_car}, {"cdr", f_cdr}, {"+", f_add}, {"-", f_sub}, 
    {"*", f_mul}, {"/", f_div}, {"int", f_int}, {"<", f_lt}, 
    {"eq?", f_eq}, {"or", f_or}, {"and", f_and}, {"not", f_not}, 
    {"cond", f_cond}, {"if", f_if}, {"let*", f_leta}, 
    {"lambda", f_lambda}, {"define", f_define}, {0}
};

f64 bind(f64 v, f64 t, f64 e) { 
    return Tk(v) == NIL ? e : 
           Tk(v) == CONS ? bind(cdr(v), cdr(t), pair(car(v), car(t), e)) : 
           pair(v, t, e); 
}

f64 reduce(f64 f, f64 t, f64 e) { 
    return eval(cdr(car(f)), bind(car(car(f)), evlis(t, e), not(cdr(f)) ? env : cdr(f))); 
}

f64 apply(f64 f, f64 t, f64 e) { 
    return Tk(f) == PRIM ? prim[ord(f)].f(t, e) : 
           Tk(f) == CLOS ? reduce(f, t, e) : err;
}

f64 eval(f64 x, f64 e) { 
    return Tk(x) == ATOM ? assoc(x, e) : 
           Tk(x) == CONS ? apply(eval(car(x), e), cdr(x), e) : x; 
}

char buf[40], see = ' ';

void look()
{
    int c = getchar();
    see = c;
    if (c == EOF)
        exit(0);
}

uint seeing(char c) { return c == ' ' ? see > 0 && see <= c : see == c; }

char get()
{
    char c = see;
    look();
    return c;
}

// 掃描器 lexer = scanner
char scan()
{
    int i = 0;
    while (seeing(' '))
        look();
    if (seeing('(') || seeing(')') || seeing('\''))
        buf[i++] = get();
    else
        do
            buf[i++] = get();
        while (i < 39 && !seeing('(') && !seeing(')') && !seeing(' '));
    return buf[i] = 0, *buf;
}

f64 Read() { return scan(), parse(); }

f64 list()
{
    f64 x;
    return scan() == ')' ? nil : 
           !strcmp(buf, ".") ? (x = Read(), scan(), x) : 
           (x = parse(), cons(x, list()));
}

f64 quote() { return cons(atom("quote"), cons(Read(), nil)); }

f64 atomic()
{
    f64 n;
    int i;
    return sscanf(buf, "%lg%n", &n, &i) > 0 && !buf[i] ? n : atom(buf);
}

f64 parse() {
    return *buf == '(' ? list() : 
           *buf == '\'' ? quote() : atomic();
}

void print(f64);

void printlist(f64 t)
{
    for (putchar('(');; putchar(' '))
    {
        print(car(t));
        if (not(t = cdr(t)))
            break;
        if (Tk(t) != CONS)
        {
            printf(" . ");
            print(t);
            break;
        }
    }
    putchar(')');
}

void print(f64 x)
{
    if (Tk(x) == NIL)
        printf("()");
    else if (Tk(x) == ATOM)
        printf("%s", (char *)cell + ord(x));
    else if (Tk(x) == PRIM)
        printf("<%s>", prim[ord(x)].s);
    else if (Tk(x) == CONS)
        printlist(x);
    else if (Tk(x) == CLOS)
        printf("{%u}", ord(x));
    else
        printf("%.10lg", x);
}

void gc() { sp = ord(env); }

int main()
{
    int i;
    printf("tinylisp");
    nil = box(NIL, 0);
    err = atom("ERR");
    tru = atom("#t");
    env = pair(tru, tru, nil);

    for (i = 0; prim[i].s; ++i) {
        env = pair(atom(prim[i].s), box(PRIM, i), env);
    }

    while (1)
    {
        printf("\n%u>", sp - hp / 8);
        print(eval(Read(), env));
        gc();
    }
}
