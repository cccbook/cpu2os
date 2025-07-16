wsl> gcc openmp_wc.c string_utilities.c -o openmp_wc -fopenmp -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -lglib-2.0
wsl> ./openmp_wc
Please give some file names on the command line.
wsl> ./openmp_wc wordcount.c
wordcount.c:    67
Σ:      67