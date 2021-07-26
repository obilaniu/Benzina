/*
** $Id: lbcachefslib.c $
** Standard I/O (and system) library
** See Copyright Notice in lua.h
*/

#define lbcachefslib_c
#define LUA_LIB

#include "lprefix.h"

#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lua.h"

#include "lauxlib.h"
#include "lualib.h"
#include "internal.h"


static int bcachefs_hello (lua_State *L) {
  printf("Hello BCacheFS\n");
  return 0;
}


/*
** functions for 'bcachefs' library
*/
static const luaL_Reg bcachefslib[] = {
  {"hello", bcachefs_hello},
  {NULL, NULL}
};


LUAMOD_API int luaopen_bcachefs (lua_State *L) {
  luaL_newlib(L, bcachefslib);  /* new module */
  return 1;
}


BENZINA_LUAOPEN_REGISTER("bcachefs", luaopen_bcachefs)
