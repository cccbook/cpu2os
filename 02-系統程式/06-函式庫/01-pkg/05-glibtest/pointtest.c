#include <glib.h>
#include "point.c"

typedef struct {
  struct point p1, p2;
} Module;

static void module_up (Module *module, gconstpointer user_data)
{
  module->p1 = (struct point) {.x=1.0,.y=2.0};
  module->p2 = (struct point) {.x=1.0,.y=2.0000001};
}

static void module_down (Module *module, gconstpointer user_data)
{
}

static void module_test1 (Module *module, gconstpointer user_data)
{
  g_assert(near(&module->p1, &module->p2));
}

int main (int argc, char *argv[]) {
  g_test_init (&argc, &argv, NULL);
  g_test_add ("/module/test1", Module, "test near()",
               module_up, module_test1, module_down);
  return g_test_run ();
}
