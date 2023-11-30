# db_tutorial

```
$ make
gcc -std=gnu11 db.c -o db

ccckmit@asus MINGW64 /d/ccc/test/db_tutorial (master)
$ make run
gcc -std=gnu11 db.c -o db
./db mydb.db
db > insert ccc
Syntax error. Could not parse statement.
db > insert ccc, 123
Syntax error. Could not parse statement.
db > select
Executed.
db > insert 1 cstack foo@bar.com
Executed.
db > select
(1, cstack, foo@bar.com)
Executed.
db > insert 2 bob bob@example.com
Executed.
db > select
(1, cstack, foo@bar.com)
(2, bob, bob@example.com)
Executed.
db > .exit
```
