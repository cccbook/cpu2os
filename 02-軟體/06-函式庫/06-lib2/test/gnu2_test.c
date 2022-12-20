#include <lib2.h>

int main() {
  double f[] = { 1.0, 2.0, 3.0, 4.0 }, *g = NEW(double, 4);

  EACH(f, 4, PUT); BR;
  MAP(f, 4, _lambda(float,(float x),{ return x*2.0; }), g);
  EACH(g, 4, PUT); BR;

  double h[4];
  MAP2(f, g, 4, fxy(float, x+y), h);
  EACH(h, 4, PUT); BR;
  FREE(g);
}
