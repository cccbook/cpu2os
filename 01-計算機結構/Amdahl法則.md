# Amdahl's Law (安達荷法則)

* [Amdahl's Law](https://chi_gitbook.gitbooks.io/personal-note/content/amdahls_law.html)
* [Amdahl's law and Gustafson's law](https://blog.csdn.net/qq_34594236/article/details/79674204)

針對電腦系統裡面某一個特定的元件予以最佳化，對於整體系統有多少的效能改變

ccc: 請用寫出 Amdahl's law 的公式，用 latex

ChatGPT:

Amdahl's law can be expressed mathematically as:

S = 1 / ( (1 - p) + (p/n) )

where:

S is the speedup of the program or system
p is the proportion of the program or system that can be parallelized (also known as the parallel fraction)
n is the number of processors

For example, if a program has a parallel fraction of 0.8 (80% of the program can be parallelized) and it is run on a system with 2 processors, the speedup would be calculated as:

S = 1 / ( (1 - 0.8) + (0.8/2) ) = 1.6

Note that the speedup is limited by the non-parallelizable portion of the program (1 - p). As the number of processors increases, the speedup will approach the maximum value of p, but it will never exceed it.