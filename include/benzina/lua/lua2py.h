/* Include Guard */
#ifndef INCLUDE_BENZINA_LUA_LUA2PY_H
#define INCLUDE_BENZINA_LUA_LUA2PY_H

/* Includes */
#include "benzina/lua/py2lua.h"

/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif

BENZINA_PUBLIC PyObject *PyObject_FromLua(lua_State *L, int index);

/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
