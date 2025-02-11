/* Compile with:
make sizesof CFLAGS="-g -Wall -std=gnu11 -O3"
*/
#include <stdio.h>
#include <math.h>

#define peval(cmd) printf(#cmd ": %g\n", cmd)

// #define pchar(ch) printf("%c\n", #@ch)

#define Setup_list(name, ...)                               \
    double *name ## _list = (double []){__VA_ARGS__, NAN};  \
    int name ## _len = 0;                                   \
    for (name ## _len =0;                                   \
            !isnan(name ## _list[name ## _len]);            \
            ) name ## _len ++;

int main(){
    double *plist = (double[]){1, 2, 3};
    double list[] = {1, 2, 3};

    // #exp : stringizing operator => "exp 運算式的字串"
    peval(sizeof(plist)/(sizeof(double)+0.0));
    peval(sizeof(list)/(sizeof(double)+0.0));

    // #@var : charizing operator => 'var 對應的字元'
    // char a, b;
    // pchar(a);
    // pchar(b);

    // ## : token parsing operator => 多個參數合成一個參數
    Setup_list(items, 1, 2, 4, 8);
    double sum=0;
    for (double *ptr= items_list; !isnan(*ptr); ptr++)
        sum += *ptr;
    printf("total for items list: %g\n", sum);

    #define Length(in) in ## _len

    sum=0;
    Setup_list(next_set, -1, 2.2, 4.8, 0.1);
    for (int i=0; i < Length(next_set); i++)
        sum += next_set_list[i];
    printf("total for next set list: %g\n", sum);
}
