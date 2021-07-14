/* Includes */
#include <Python.h>
#include "lauxlib.h"
#include "benzina/lua/lua2py.h"
#include "benzina/lua/py2lua.h"

/* Defines */

/* Functions */

/* Main */

int main()
{
    lua_State *L = luaL_newstate();
    Py_Initialize();

    PyObject *test_bool = PyBool_FromLong((long)0);
    PyObject *test_bytes = PyBytes_FromString("test bytes");
    PyObject *test_float = PyFloat_FromDouble((double)4.7);
    PyObject *test_long = PyLong_FromLong((long)9);
    PyObject *test_none = Py_None;
    PyObject *test_str = PyUnicode_FromString("test str");
    PyObject *test_dict = PyDict_New();
    PyObject *test_list = PyList_New(0);
    assert(PyDict_SetItem(test_dict, test_bool, test_bool) == 0);
    assert(PyDict_SetItem(test_dict, test_bytes, test_bytes) == 0);
    assert(PyDict_SetItem(test_dict, test_float, test_float) == 0);
    assert(PyDict_SetItem(test_dict, test_long, test_long) == 0);
    assert(PyDict_SetItem(test_dict, test_none, test_none) == 0);
    assert(PyDict_SetItem(test_dict, test_str, test_str) == 0);
    assert(PyDict_SetItemString(test_dict, "key list", test_list) == 0);
    PyObject *test_dict_cp = PyDict_Copy(test_dict);
    assert(PyDict_SetItemString(test_dict, "key dict", test_dict_cp) == 0);
    assert(PyList_Append(test_list, test_bool) == 0);
    assert(PyList_Append(test_list, test_bytes) == 0);
    assert(PyList_Append(test_list, test_float) == 0);
    assert(PyList_Append(test_list, test_long) == 0);
    assert(PyList_Append(test_list, test_none) == 0);
    assert(PyList_Append(test_list, test_str) == 0);

    assert(lua_pushpyobject(L, test_bool) == 1);
    assert(lua_toboolean(L, -1) == (test_bool == Py_True));
    lua_pop(L, 1);
    assert(lua_gettop(L) == 0);

    assert(lua_pushpyobject(L, test_bytes) == 1);
    assert(strcmp(lua_tostring(L, -1), PyBytes_AsString(test_bytes)) == 0);
    lua_pop(L, 1);
    assert(lua_gettop(L) == 0);

    assert(lua_pushpyobject(L, test_float) == 1);
    assert(lua_tonumber(L, -1) - PyFloat_AsDouble(test_float) == 0.0);
    lua_pop(L, 1);
    assert(lua_gettop(L) == 0);

    assert(lua_pushpyobject(L, test_long) == 1);
    assert(lua_tointeger(L, -1) == PyLong_AsLong(test_long));
    lua_pop(L, 1);
    assert(lua_gettop(L) == 0);

    assert(lua_pushpyobject(L, test_none) == 1);
    assert(strcmp(lua_tostring(L, -1), LUA_NONE) == 0);
    lua_pop(L, 1);
    assert(lua_gettop(L) == 0);

    assert(lua_pushpyobject(L, test_str) == 1);
    assert(strcmp(lua_tostring(L, -1), PyUnicode_AsUTF8(test_str)) == 0);
    lua_pop(L, 1);
    assert(lua_gettop(L) == 0);

    int d = lua_pushpyobject(L, test_dict);
    int len = 0;
    lua_pushnil(L);
    for (len = 0; lua_next(L, -2) != 0; ++len) { lua_pop(L, 1); }
    assert(len == PyObject_Length(test_dict));

    lua_pushpyobject(L, test_bool);
    lua_gettable(L, d);
    assert(lua_toboolean(L, -1) == (test_bool == Py_True));
    lua_pop(L, 1);
    lua_pushpyobject(L, test_bytes);
    lua_gettable(L, d);
    assert(strcmp(lua_tostring(L, -1), PyBytes_AsString(test_bytes)) == 0);
    lua_pop(L, 1);
    lua_pushpyobject(L, test_float);
    lua_gettable(L, d);
    assert(lua_tonumber(L, -1) - PyFloat_AsDouble(test_float) == 0.0);
    lua_pop(L, 1);
    lua_pushpyobject(L, test_long);
    lua_gettable(L, d);
    assert(lua_tointeger(L, -1) == PyLong_AsLong(test_long));
    lua_pop(L, 1);
    lua_pushpyobject(L, test_none);
    lua_gettable(L, d);
    assert(strcmp(lua_tostring(L, -1), LUA_NONE) == 0);
    lua_pop(L, 1);
    lua_pushpyobject(L, test_str);
    lua_gettable(L, d);
    assert(strcmp(lua_tostring(L, -1), PyUnicode_AsUTF8(test_str)) == 0);
    lua_pop(L, 1);
    lua_pushstring(L, "key list");
    lua_gettable(L, d);
    assert(luaL_len(L, -1) == PyObject_Length(test_list));
    lua_pop(L, 1);
    lua_pushstring(L, "key dict");
    lua_gettable(L, d);
    lua_pushnil(L);
    for (len = 0; lua_next(L, -2) != 0; ++len) { lua_pop(L, 1); }
    assert(len == PyObject_Length(test_dict_cp));
    lua_pop(L, 1);

    lua_pop(L, 1);
    assert(lua_gettop(L) == 0);

    int t = lua_pushpyobject(L, test_list);
    assert(luaL_len(L, t) == PyObject_Length(test_list));

    assert(lua_rawgeti(L, t, 1) == LUA_TBOOLEAN);
    assert(lua_toboolean(L, -1) == (test_bool == Py_True));
    lua_pop(L, 1);
    assert(lua_rawgeti(L, t, 2) == LUA_TSTRING);
    assert(strcmp(lua_tostring(L, -1), PyBytes_AsString(test_bytes)) == 0);
    lua_pop(L, 1);
    assert(lua_rawgeti(L, t, 3) == LUA_TNUMBER);
    assert(lua_tonumber(L, -1) - PyFloat_AsDouble(test_float) == 0.0);
    lua_pop(L, 1);
    assert(lua_rawgeti(L, t, 4) == LUA_TNUMBER);
    assert(lua_tointeger(L, -1) == PyLong_AsLong(test_long));
    lua_pop(L, 1);
    assert(lua_rawgeti(L, t, 5) == LUA_TSTRING);
    assert(strcmp(lua_tostring(L, -1), "__NONE__") == 0);
    lua_pop(L, 1);
    assert(lua_rawgeti(L, t, 6) == LUA_TSTRING);
    assert(strcmp(lua_tostring(L, -1), PyUnicode_AsUTF8(test_str)) == 0);
    lua_pop(L, 1);

    lua_pop(L, 1);
    assert(lua_gettop(L) == 0);
    PyObject *o = NULL;

    assert(lua_pushpyobject(L, test_bool) == 1);
    o = PyObject_FromLua(L, -1);
    assert(o == Py_False);
    lua_pop(L, 1);
    Py_DECREF(o);

    assert(lua_pushpyobject(L, test_bytes) == 1);
    o = PyObject_FromLua(L, -1);
    assert(strcmp(lua_tostring(L, -1), PyBytes_AsString(o)) == 0);
    lua_pop(L, 1);
    Py_DECREF(o);

    assert(lua_pushpyobject(L, test_float) == 1);
    o = PyObject_FromLua(L, -1);
    assert(lua_tonumber(L, -1) - PyFloat_AsDouble(o) == 0.0);
    lua_pop(L, 1);
    Py_DECREF(o);

    assert(lua_pushpyobject(L, test_long) == 1);
    o = PyObject_FromLua(L, -1);
    assert(lua_tointeger(L, -1) == PyLong_AsLong(o));
    lua_pop(L, 1);
    Py_DECREF(o);

    assert(lua_pushpyobject(L, test_none) == 1);
    o = PyObject_FromLua(L, -1);
    assert(o == Py_None);
    lua_pop(L, 1);
    Py_DECREF(o);

    assert(lua_pushpyobject(L, test_str) == 1);
    o = PyObject_FromLua(L, -1);
    assert(strcmp(lua_tostring(L, -1), PyBytes_AsString(o)) == 0);
    lua_pop(L, 1);
    Py_DECREF(o);

    return 0;
}
