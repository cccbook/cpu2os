//ex‑gtree‑1.c
#include <stdio.h>
#include <glib.h>
int main(int argc, char** argv) {
 GTree* t = g_tree_new((GCompareFunc)g_ascii_strcasecmp);
 g_tree_insert(t, "c", "Chicago");
 printf("The tree height is %d because there's only one node\n", g_tree_height(t));
 g_tree_insert(t, "b", "Boston");
 g_tree_insert(t, "d", "Detroit");
 printf("Height is %d since c is root; b and d are children\n", g_tree_height(t));
 printf("There are %d nodes in the tree\n", g_tree_nnodes(t));
 g_tree_remove(t, "d");
 printf("After remove(), there are %d nodes in the tree\n", g_tree_nnodes(t));
 g_tree_destroy(t);
 return 0;
}
/*
***** Output *****

The tree height is 1 because there's only one node
Height is 2 since c is root; b and d are children
There are 3 nodes in the tree
After remove(), there are 2 nodes in the tree
*/
