#include "benzina/benzina.h"
#include "benzina/visibility.h"
#include <stdlib.h>

extern "C" __attribute__((visibility("default"))) int foo(const char* p){
    return strtoul(p, 0, 0);
}
