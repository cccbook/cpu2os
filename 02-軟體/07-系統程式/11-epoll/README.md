# epoll 範例

epoll1.c 來源 -- https://suchprogramming.com/epoll-in-3-easy-steps/

```
guest@localhost:~/sp/09-linux/01-c/03-epoll$ gcc epoll1.c -o epoll1
guest@localhost:~/sp/09-linux/01-c/03-epoll$ ./epoll1

Polling for input...
hello
1 ready events
Reading file descriptor '0' -- 6 bytes read.
Read 'hello
'

Polling for input...
hello how are you, i am ccc, he is snoopy
1 ready events
Reading file descriptor '0' -- 10 bytes read.
Read 'hello how '

Polling for input...
1 ready events
Reading file descriptor '0' -- 10 bytes read.
Read 'are you, i'

Polling for input...
1 ready events
Reading file descriptor '0' -- 10 bytes read.
Read ' am ccc, h'

Polling for input...
1 ready events
Reading file descriptor '0' -- 10 bytes read.
Read 'e is snoop'

Polling for input...
1 ready events
Reading file descriptor '0' -- 2 bytes read.
Read 'y
'

Polling for input...

1 ready events
Reading file descriptor '0' -- 1 bytes read.
Read '
'

Polling for input...

1 ready events
Reading file descriptor '0' -- 1 bytes read.
Read '
'

Polling for input...
 
1 ready events
Reading file descriptor '0' -- 2 bytes read.
Read '
'

Polling for input...

1 ready events
Reading file descriptor '0' -- 1 bytes read.
Read '
'

Polling for input...

1 ready events
Reading file descriptor '0' -- 1 bytes read.
Read '
'

Polling for input...
hello
1 ready events
Reading file descriptor '0' -- 6 bytes read.
Read 'hello
'

Polling for input...
hi
1 ready events
Reading file descriptor '0' -- 3 bytes read.
Read 'hi
'

Polling for input...
^[[A
1 ready events
Reading file descriptor '0' -- 4 bytes read.
'ead '

Polling for input...
hello how are you
1 ready events
Reading file descriptor '0' -- 10 bytes read.
Read 'hello how '

Polling for input...
1 ready events
Reading file descriptor '0' -- 8 bytes read.
Read 'are you
'

Polling for input...
hello how are you .... i am snoopy
1 ready events
Reading file descriptor '0' -- 10 bytes read.
Read 'hello how '

Polling for input...
1 ready events
Reading file descriptor '0' -- 10 bytes read.
Read 'are you ..'

Polling for input...
1 ready events
Reading file descriptor '0' -- 10 bytes read.
Read '.. i am sn'

Polling for input...
1 ready events
Reading file descriptor '0' -- 5 bytes read.
Read 'oopy
'

Polling for input...
0 ready events

Polling for input...
```