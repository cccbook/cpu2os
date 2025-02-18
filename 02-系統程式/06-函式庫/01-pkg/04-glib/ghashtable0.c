// https://gist.github.com/tristanwietsma/6110290

#include <stdio.h>
#include <glib.h>

int main() {

	// load string hash table
	GHashTable* sm = g_hash_table_new(g_str_hash, g_str_equal);
	g_hash_table_insert(sm, "a", "alfa");

	// lookup key
	printf("a => %s\n", (char*) g_hash_table_lookup(sm, "a"));

	// replace a value
	g_hash_table_replace(sm, "a", "ALFA");
	printf("a => %s\n", (char*) g_hash_table_lookup(sm, "a"));

	// free memory
	g_hash_table_destroy(sm);
    return 0;
}