gcc -Wall -Wextra -g -fsanitize=address -fsanitize=undefined compiler.c -o compiler
gcc -Wall -Wextra -g -fsanitize=address -fsanitize=undefined vm.c -o vm
gcc -Wall -Wextra -g -fsanitize=address -fsanitize=undefined ir2asm.c -o ir2asm