// https://hackmd.io/@CynthiaChuang/Regular-Expressions-in-C
#include <stdio.h>
#include <stdbool.h>
#include <regex.h>  
#include <assert.h>

int main(){
    regex_t preg; // 宣告編譯結果變數
    const char* pattern = "[a-z0-9_]+@([a-z0-9-]+\\.)+[a-z0-9]+"; // 定義表示式
    int success = regcomp(&preg, pattern, REG_EXTENDED|REG_ICASE); // 編譯，這邊使用 ERE，且不考慮大小寫
    assert(success==0);

    char* target = "name:ccc email:ccckmit@gmail.com  age:50";   //目標字串
    regmatch_t matchptr[1];   // 記錄匹配結果陣列，長度為1僅記錄 full match
    const size_t nmatch = 1;    //  matchptr陣列長度
    int status = regexec(&preg, target, nmatch, matchptr, 0); //匹配
    if (status == REG_NOMATCH){ // 沒匹配
        printf("No Match\n");
    }
    else if (status == 0){  // 匹配
        printf("Match\n");
        // 取出起始與結束位置印出字串
        for (int i = matchptr[0].rm_so; i < matchptr[0].rm_eo; i++){  
            printf("%c", target[i]);
        }
        printf("\n");
    }
    else {  // 執行錯誤
        char msgbuf[256];
        regerror(status, &preg, msgbuf, sizeof(msgbuf)); 
        printf("error: %s\n", msgbuf);
    }

    regfree(&preg);  // 釋放
    return 0;
}