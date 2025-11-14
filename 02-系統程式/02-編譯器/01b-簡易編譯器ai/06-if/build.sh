gcc -Wall -Wextra -g -fsanitize=address -fsanitize=undefined ir.c compiler.c -o compiler
gcc -Wall -Wextra -g -fsanitize=address -fsanitize=undefined ir.c vm.c -o vm
gcc -Wall -Wextra -g -fsanitize=address -fsanitize=undefined ir.c ir2asm.c -o ir2asm
