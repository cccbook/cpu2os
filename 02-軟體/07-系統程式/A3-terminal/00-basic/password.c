#include <termios.h>
#include <stdio.h>
#include <stdlib.h>

#define PASSWORD_LEN 8

int main() {
    struct termios initialresettings, newresettings;
    char password[PASSWORD_LEN + 1];
    tcgetattr(fileno(stdin), & initialresettings);
    newresettings = initialresettings;
    newresettings.c_lflag &= ~ECHO;
    printf("Enter password: ");
    if (tcsetattr(fileno(stdin), TCSAFLUSH, &newresettings) !=0) { // TCSAFLUSH: 輸出完成後改變
        fprintf(stderr, "Could not set attributes\n");
    } else {
        fgets(password, PASSWORD_LEN, stdin);
        tcsetattr(fileno(stdin), TCSANOW, &initialresettings); // TCSANOW: 立即改變
        fprintf(stdout, "\nYou Entered %s\n", password);
    }
    exit(0);
}