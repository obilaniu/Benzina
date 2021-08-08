/* Includes */
#include <Python.h>
#include "benzina/lua/lua2py.h"
#include "lauxlib.h"

BENZINA_PUBLIC PyObject *PyObject_FromLua(lua_State *L, int index){
    assert(lua_type(L, index));

    if(index < 0){
        index = lua_gettop(L) + index + 1;
    }
    PyObject *o = NULL;

    switch (lua_type(L, index)){
    case LUA_TBOOLEAN:
        o = lua_toboolean(L, index) ? Py_True : Py_False;
        break;
    case LUA_TSTRING:
        if(strcmp(lua_tostring(L, -1), LUA_NONE) == 0){
            o = Py_None;
        }else{
            o = PyBytes_FromString(lua_tostring(L, index));
        }
        break;
    case LUA_TNUMBER:
        if((double)lua_tointeger(L, index) == lua_tonumber(L, index)){
            o = PyLong_FromLong(lua_tointeger(L, index));
        }else{
            o = PyFloat_FromDouble(lua_tonumber(L, index));
        }
        break;
    case LUA_TTABLE:
        if(lua_rawlen(L, index)){
            o = PyList_New((long)lua_rawlen(L, index));
            lua_pushnil(L);
            for(int i = 0; lua_next(L, index) != 0; ++i){
                PyObject *v = PyObject_FromLua(L, -1);
                PyList_Insert(o, i, v);
                Py_DECREF(v);
                lua_pop(L, 1);
            }
        }else{
            o = PyDict_New();
            lua_pushnil(L);
            for(int i = 0; lua_next(L, index) > 0; ++i){
                PyObject *k = PyObject_FromLua(L, -2);
                PyObject *v = PyObject_FromLua(L, -1);
                PyDict_SetItem(o, k, v);
                Py_DECREF(k);
                Py_DECREF(v);
                lua_pop(L, 1);
            }
        }
        break;
    }

    return o;
}
