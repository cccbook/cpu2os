(py310) cccimac@cccimacdeiMac A2-sq0 % make      
gcc -Wall -Wextra -std=c99 -g -c main.c -o main.o
gcc -Wall -Wextra -std=c99 -g -c sq0.c -o sq0.o
gcc -Wall -Wextra -std=c99 -g -c pager.c -o pager.o
gcc -Wall -Wextra -std=c99 -g -c btree.c -o btree.o
gcc -Wall -Wextra -std=c99 -g -c parser.c -o parser.o
gcc -Wall -Wextra -std=c99 -g -c vm.c -o vm.o
gcc -Wall -Wextra -std=c99 -g -c util.c -o util.o
gcc -Wall -Wextra -std=c99 -g -c lexer.c -o lexer.o
gcc -Wall -Wextra -std=c99 -g main.o sq0.o pager.o btree.o parser.o vm.o util.o lexer.o -o sq0_db
(py310) cccimac@cccimacdeiMac A2-sq0 % make test
gcc -Wall -Wextra -std=c99 -g -c test_sq0.c -o test_sq0.o
gcc -Wall -Wextra -std=c99 -g sq0.o pager.o btree.o parser.o vm.o util.o lexer.o test_sq0.o -o run_tests
--- Running Tests ---
./run_tests
--- Running Test 1: Insert and Select ---

--- Results ---
(1, user1, user1@example.com)
(2, user2, user2@example.com)
(100, max_id_user, max@id.com)
---------------
--- Test 1 Finished ---
--- Running Test 2: Persistence ---
Test Failed: test_sq0.c, Line 129: After reopening, row count must still be 3
Test Failed: test_sq0.c, Line 135: After inserting a new row, count must be 4
--- Test 2 Finished ---
--- Running Test 3: Key Collision ---
Test Failed: test_sq0.c, Line 157: Row count must not change on key collision
--- Test 3 Finished ---

==================================
Tests Run: 6, Failed: 3
==================================
make: *** [test] Error 1