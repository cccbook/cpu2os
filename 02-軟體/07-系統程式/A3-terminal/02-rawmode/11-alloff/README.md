# kilo

```
mac020:11-alloff mac020$ make
cc kilo.c -o kilo -Wall -Wextra -pedantic -std=c99
mac020:11-alloff mac020$ ./kilo
97 ('a')
        100 ('d')
                 102 ('f')
                          106 ('j')
                                   108 ('l')
                                            97 ('a')
                                                    100 ('d')
                                                             102 ('f')
                                                                      97 ('a')
                                                                              100 ('d')
                                                                                      108 ('l')
        102 ('f')
                 106 ('j')
                          99 ('c')
                                  3
                                   26
                                     24
                                       22
                                         13
                                           mac020:11-alloff mac020$ 
```
