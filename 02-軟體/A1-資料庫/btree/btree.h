#ifndef _BTREE_H_
#define _BTREE_H_

#define T 100
#define BTREE_OK 0
#define BTREE_ERR -1

typedef struct btree_node
{
     int key_num;
     int key[2 * T - 1];
     int seek[2 * T];
     int self;
     int parent;
}btree_node;

typedef struct btree
{
     struct btree_node *root;
     FILE *fp;
}btree;
int btree_insert(btree *tree, int key);
int btree_split(btree *tree, btree_node *node);
btree_node *btree_search(btree *tree, int key);
btree *btree_create(const char *file);
int btree_delete(btree *tree, int key);
int btree_merge(btree *tree, btree_node *left, btree_node *right, btree_node *parent);
#endif
