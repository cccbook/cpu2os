#pragma once

#include <math.h>
#include <stdbool.h>

struct point {
  double x, y;
};

struct point diff(const struct point *p1, const struct point *p2);
int near(const struct point *p1, const struct point *p2);
