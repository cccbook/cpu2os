# ghashtable

```
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh glib_hashtable1
gcc: error: glib_hashtable1.c: No such file or directory
./build.sh: line 2: ./glib_hashtable1: No such file or directory
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh ghashtable1
gcc: error: ghashtable1.c: No such file or directory
./build.sh: line 2: ./ghashtable1: No such file or directory
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh ghashtable0
a => alfa
a => ALFA
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh ghashtable1lookup
There are 3 keys in the hash
The capital of Texas is Austin
The value 'Virginia' was found and removed
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh ghashtable2
Calling insert with the texas_2 key
Got a key destroy call for texas_2
Calling replace with the texas_2 key
Got a key destroy call for texas_1
Destroying hash, so goodbye texas_2
Got a key destroy call for texas_2
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh ghashtable3each
The square of 1 is one
The square of 2 is four
The square of 3 is nine
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh ghashtable4
36 + 6 == 42
Got a value destroy call for 12
Got a value destroy call for 36
Got a value destroy call for 22
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh ghashtable5
Removing New York, you should see two callbacks
Got a GDestroyNotify callback
Got a GDestroyNotify callback
Texas has been stolen, 3 items remaining
Stealing remaining items
Destroying the GHashTable, but it's empty, so no callbacks
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh ghashtable6
Found that the capital of Columbus is Ohio
Couldn't find Vermont in the hash table
guest@localhost:~/cpu2os/02-軟體/06-函式庫/01-pkg/04-glib$ ./build.sh ghashtable7
Here are some cities in Texas: Austin Houston
Here are some cities in Virginia: Richmond Keysville
Freeing a GSList, first item is Austin
Freeing a GSList, first item is Richmond
```