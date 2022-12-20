#include "point.h"

struct point diff(const struct point *p1, const struct point *p2) {
    struct point d;
    d.x=p1->x-p2->x;
    d.y=p1->y-p2->y;
    return d; 
}

int near(const struct point *p1, const struct point *p2) {
    struct point d = diff(p1, p2);
    return fabs(d.x) < 0.01 && fabs(d.y) < 0.01; // 0.00000000000001;
}