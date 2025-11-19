#include <stdio.h>
#include "btree.h"

#define N 10000

int main() {
  btree *t = btree_create("btree.data");
  for (int key=0; key<N; key++) {
    btree_insert(t, key*10);
  }
  btree_node *n1 = btree_search(t, 80);
  printf("n1=%p\n", n1);
  btree_node *n2 = btree_search(t, 81);
  printf("n2=%p\n", n2);
  for (int key=0; key<N; key++) {
    btree_delete(t, key*10);
  }
}

/*
int main(int argc, char *argv[])
{
     int i;
     btree *tree = btree_create("source");
     for(i = 0; i < 100000; i++)
          btree_insert(tree, i);
     for(i = 0; i < 100000; i++)
          btree_delete(tree, i);
     return 0;
}
*/