/* Include Guard */
#ifndef INCLUDE_BENZINA_LUA_PY2LUA_H
#define INCLUDE_BENZINA_LUA_PY2LUA_H

/* Includes */
#include "benzina/visibility.h"

/* Defines */
#define LUA_NONE "__NONE__"
typedef struct PyObject PyObject;
typedef struct lua_State lua_State;

/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif

BENZINA_PUBLIC int lua_pushpyobject(lua_State *L, PyObject *o);

/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
