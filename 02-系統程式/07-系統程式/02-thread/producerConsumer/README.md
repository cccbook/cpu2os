# producerConsumer.c

```
$ gcc -std=c11 producerConsumer.c -o producerConsumer -lpth
read

$ ./producerConsumer
Producer: produce 41, buffer has 1 data
Consumer: consume 41, buffer has 0 data
Producer: produce 34, buffer has 1 data
Consumer: consume 34, buffer has 0 data
Producer: produce 69, buffer has 1 data
Consumer: consume 69, buffer has 0 data
Producer: produce 78, buffer has 1 data
Producer: produce 62, buffer has 2 data
Consumer: consume 78, buffer has 1 data
Producer: produce 5, buffer has 2 data
Consumer: consume 62, buffer has 1 data
Producer: produce 81, buffer has 2 data
Producer: produce 61, buffer has 3 data
Consumer: consume 5, buffer has 2 data
```

