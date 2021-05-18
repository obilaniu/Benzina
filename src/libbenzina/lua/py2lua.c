/* Includes */
#include <Python.h>
#include "benzina/lua/py2lua.h"
#include "lauxlib.h"

// Translation Python -> Lua
//   bool -> bool
//   bytes -> bytes
//   float -> number
//   int -> integer
//   str -> string
//   dict -> table
//   list -> table
//   None -> "__NONE__"

BENZINA_PUBLIC int lua_pushpyobject(lua_State *L, PyObject *o){
    PyObject *key = NULL, *value = NULL;
    Py_ssize_t pos = 0;
    Py_ssize_t size = 0;

    int top = lua_gettop(L) + 1;

    if(PyBool_Check(o)){
        lua_pushboolean(L, o == Py_True);
    }else if(PyBytes_Check(o)){
        lua_pushstring(L, PyBytes_AsString(o));
    }else if(PyFloat_Check(o)){
        lua_pushnumber(L, PyFloat_AsDouble(o));
    }else if(PyLong_Check(o)){
        lua_pushinteger(L, PyLong_AsLong(o));
    }else if(PyUnicode_Check(o)){
        lua_pushstring(L, PyUnicode_AsUTF8(o));
    }else if(PyDict_Check(o)){
        size = PyDict_Size(o);
        lua_createtable(L, 0, (int)size);                   // t: -3
        while(PyDict_Next(o, &pos, &key, &value)){
            lua_pushpyobject(L, key);                       // k: -2
            lua_pushpyobject(L, value);                     // k: -1
            lua_settable(L, -3);
        }
        printf("%d\n", (int)luaL_len(L, -1));
        printf("r:%d\n", (int)lua_rawlen(L, -1));
    }else if(PyList_Check(o)){
        size = PyList_Size(o);
        lua_createtable(L, (int)size, 0);                   // t: -2
        for(Py_ssize_t idx = 0; idx < size; idx++){
            lua_pushpyobject(L, PyList_GetItem(o, idx));    // v: -1
            lua_rawseti(L, -2, idx+1);
        }
        printf("%d\n", (int)luaL_len(L, -1));
        printf("r:%d\n", (int)lua_rawlen(L, -1));
    }else if(o == Py_None){
        lua_pushstring(L, LUA_NONE);
    }else{
        top = 0;
    }

    return top;
}
