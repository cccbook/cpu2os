gcc -Wall -Wextra -g -fsanitize=address -fsanitize=undefined compiler.c -o compiler
gcc -Wall -Wextra -g -fsanitize=address -fsanitize=undefined vm.c -o vm