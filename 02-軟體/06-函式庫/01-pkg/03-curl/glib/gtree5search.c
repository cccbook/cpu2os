//ex‑gtree‑5.c
#include <stdio.h>
#include <glib.h>
gint finder(gpointer key, gpointer user_data) {
int len = strlen((char*)key);
 if (len == 3) {
  return 0;
 }
 return (len<3)?1:-1;
}
int main(int argc, char** argv) {
 GTree* t = g_tree_new((GCompareFunc)g_ascii_strcasecmp);
 g_tree_insert(t, "dddd", "Detroit");
 g_tree_insert(t, "a", "Annandale");
 g_tree_insert(t, "ccc", "Cleveland");
 g_tree_insert(t, "bb", "Boston");
 gpointer value = g_tree_search(t, (GCompareFunc)finder, NULL);
 printf("Located value %s; its key is 3 characters long\n", (char*) value);
 g_tree_destroy(t);
 return 0;
}

/*
***** Output *****

Located value Cleveland; its key is 3 characters long
*/
