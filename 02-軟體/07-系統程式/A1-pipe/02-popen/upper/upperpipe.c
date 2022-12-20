#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    char line[8192];
    FILE *fpin;
    fpin = popen("./upper", "r");
    for ( ; ; ) {
        fputs("prompt> ", stdout);
        fflush(stdout);
        fgets(line, sizeof(line), fpin); // 注意這行，line 會被 upper 轉為大寫
        fputs(line, stdout);
    }
}