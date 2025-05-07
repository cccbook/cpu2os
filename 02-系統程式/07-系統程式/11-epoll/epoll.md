# epoll

* [我读过的最好的epoll讲解](https://zhuanlan.zhihu.com/p/36764771) (讚)
    * 備份 -- https://juejin.cn/post/6963589249463500831
    * https://www.cnblogs.com/liushui-sky/p/7777665.html

輪詢 poll 浪費時間

为了避免CPU空转，可以引进了一个代理（一开始有一位叫做select的代理，后来又有一位叫做poll的代理，不过两者的本质是一样的）。这个代理比较厉害，可以同时观察许多流的I/O事件，在空闲的时候，会把当前线程阻塞掉，当有一个或多个流有I/O事件时，就从阻塞态中醒来，于是我们的程序就会轮询一遍所有的流（于是我们可以把“忙”字去掉了）。

epoll可以理解为event poll，不同于忙轮询和无差别轮询，epoll之会把哪个流发生了怎样的I/O事件通知我们。此时我们对这些流的操作都是有意义的。（复杂度降低到了O(k)，k为产生I/O事件的流的个数，也有认为O(1)的[更新 1]）

epoll_ctl(epollfd, EPOLL_CTL_ADD, socket, EPOLLIN);//有缓冲区内有数据时epoll_wait返回

epoll_ctl(epollfd, EPOLL_CTL_DEL, socket, EPOLLOUT);//缓冲区可写入时epoll_wait返回

* https://suchprogramming.com/epoll-in-3-easy-steps/
* https://github.com/onestraw/epoll-example
* https://copyconstruct.medium.com/the-method-to-epolls-madness-d9d2d6378642 (讚)

Why epoll is more performant that select and poll
As stated in the previous post, the cost of select/poll is O(N), which means when N is very large (think of a web server handling tens of thousands of mostly sleepy clients), every time select/poll is called, even if there might only be a small number of events that actually occurred, the kernel still needs to scan every descriptor in the list.
Since epoll monitors the underlying file description, every time the open file description becomes ready for I/O, the kernel adds it to the ready list without waiting for a process to call epoll_wait to do this. When a process does call epoll_wait, then at that time the kernel doesn’t have to do any additional work to respond to the call, but instead returns all the information about the ready list it’s been maintaining all along.
Furthermore, with every call to select/poll requires passing the kernel the information about the descriptors we want to monitored. This is obvious from the signature to both calls. The kernel returns the information about all the file descriptors passed in which the process again needs to examine (by scanning all the descriptors) to find out which ones are ready for I/O.

* https://stackoverflow.com/questions/27247/could-you-recommend-some-guides-about-epoll-on-linux
* https://web.archive.org/web/20120514073512/https://banu.com/blog/2/how-to-use-epoll-a-complete-example-in-c/epoll-example.c

* https://man7.org/linux/man-pages/man7/epoll.7.html

```
CONFORMING TO
       The epoll API is Linux-specific.  Some other systems provide
       similar mechanisms, for example, FreeBSD has kqueue, and Solaris
       has /dev/poll.
```


Example:

```
  While the usage of epoll when employed as a level-triggered
       interface does have the same semantics as poll(2), the edge-
       triggered usage requires more clarification to avoid stalls in
       the application event loop.  In this example, listener is a
       nonblocking socket on which listen(2) has been called.  The
       function do_use_fd() uses the new ready file descriptor until
       EAGAIN is returned by either read(2) or write(2).  An event-
       driven state machine application should, after having received
       EAGAIN, record its current state so that at the next call to
       do_use_fd() it will continue to read(2) or write(2) from where it
       stopped before.
```

Code: 

```cpp
           #define MAX_EVENTS 10
           struct epoll_event ev, events[MAX_EVENTS];
           int listen_sock, conn_sock, nfds, epollfd;

           /* Code to set up listening socket, 'listen_sock',
              (socket(), bind(), listen()) omitted. */

           epollfd = epoll_create1(0);
           if (epollfd == -1) {
               perror("epoll_create1");
               exit(EXIT_FAILURE);
           }

           ev.events = EPOLLIN;
           ev.data.fd = listen_sock;
           if (epoll_ctl(epollfd, EPOLL_CTL_ADD, listen_sock, &ev) == -1) {
               perror("epoll_ctl: listen_sock");
               exit(EXIT_FAILURE);
           }

           for (;;) {
               nfds = epoll_wait(epollfd, events, MAX_EVENTS, -1);
               if (nfds == -1) {
                   perror("epoll_wait");
                   exit(EXIT_FAILURE);
               }

               for (n = 0; n < nfds; ++n) {
                   if (events[n].data.fd == listen_sock) {
                       conn_sock = accept(listen_sock,
                                          (struct sockaddr *) &addr, &addrlen);
                       if (conn_sock == -1) {
                           perror("accept");
                           exit(EXIT_FAILURE);
                       }
                       setnonblocking(conn_sock);
                       ev.events = EPOLLIN | EPOLLET;
                       ev.data.fd = conn_sock;
                       if (epoll_ctl(epollfd, EPOLL_CTL_ADD, conn_sock,
                                   &ev) == -1) {
                           perror("epoll_ctl: conn_sock");
                           exit(EXIT_FAILURE);
                       }
                   } else {
                       do_use_fd(events[n].data.fd);
                   }
               }
           }
```