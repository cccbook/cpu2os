# Variable Length Array

```c
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int s=1, i, temp=0, j;
    printf ("Please enter the size of array\n") ;
    scanf ("%d",&s);

    // https://www.geeksforgeeks.org/variable-length-arrays-in-c-and-c/
    // 這種 variable length array 在 C99, C11 中有支援了！
    int array[s];
    printf("sizeof(array)=%lu\n", sizeof(array));
    printf("Please enter the contents of array:\n");
    for (i=0;i<s;i++){
        scanf("%d",&array[i]);
    }
    printf("Current number \n");
    for (i=0;i<s;i++){
        printf("%d\n",array[i]);
    }
     printf("Sorted number\n");
    for(j=0;j<s;j++){
        for(i=0;i<s-1;i++){
            if(array[i] > array[i+1]){
                temp=array[i];
                array[i]=array[i+1];
                array[i+1]=temp;
            }
        }

        for(i=0;i<s;i++) printf("%d\t", array[i]);
        printf("\n");
    }
    return 0;
}
```

執行結果

```
csienqu-teacher:array csienqu$ ./arrayArrangement1.o
Please enter the size of array
3
sizeof(array)=12
Please enter the contents of array:
4 1 2
Current number 
4
1
2
Sorted number
1       2       4
1       2       4
1       2       4

csienqu-teacher:array csienqu$ ./arrayArrangement1.o
Please enter the size of array
5
sizeof(array)=20
Please enter the contents of array:
4 2 3 5 1
Current number 
4
2
3
5
1
Sorted number
2       3       4       1       5
2       3       1       4       5
2       1       3       4       5
1       2       3       4       5
1       2       3       4       5
```

