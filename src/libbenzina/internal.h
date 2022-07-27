/* Include Guard */
#ifndef SRC_LIBBENZINA_INTERNALS_H
#define SRC_LIBBENZINA_INTERNALS_H



/* This header is only usable if building libbenzina itself! */
#if !BENZINA_IS_BUILDING
# error \
    "This file cannot be meaningfully included outside of libbenzina's " \
    "own source code. It includes declarations that are made invisible " \
    "to linkage from outside the ELF shared library."
#endif



/* Includes */
#include <lua.h>
#include <lauxlib.h>
#include "benzina/benzina.h"



/**
 * @brief "Magic" macro definitions
 * 
 * The linker script defines a number of magic sections in which data may be
 * placed. These sections are then concatenated at link-time. By relying on
 * the linker to assemble decoupled Benzina modules, we reduce the number of
 * files that must be simultaneously modified to add a new module.
 * 
 *   - BENZINA_LUAOPEN_REGISTER(name, func): Registers a Lua module with the
 *     Lua searcher for statically-linked modules. If module "name" is
 *     requested, the associated func(lua_State* L) function is called.
 *   - BENZINA_TOOL_REGISTER(name, func): Registers a tool's main() entry
 *     point with the libbenzina main() entry point to enable dispatch. If
 *     tool "name" is requested, the associated int main(int argc, char* argv[])
 *     function is called.
 * 
 * The other linker constructed arrays don't have macros to add additional
 * entries because for them the optimal solution is unlikely to be within the
 * C code. For instance, if you wish to add a Lua source file and package,
 * it's probably better to use the build system engineering to add it, so that
 * it can benefit from LZ4 compression.
 */

#define ____BENZINA_REGISTER(a,b) a ## b
#define ___BENZINA_REGISTER(a,b)  ____BENZINA_REGISTER(a,b)
#define __BENZINA_REGISTER(a)     ___BENZINA_REGISTER(a, __COUNTER__)
#define BENZINA_LUAOPEN_REGISTER(name, func)                      \
    BENZINA_ATTRIBUTE_USED                                        \
    BENZINA_ATTRIBUTE_SECTION(ASM_SECT_LUAOPENARRAY_NAME)         \
    static const luaL_Reg __BENZINA_REGISTER(__benz_luaopen_entry_) = {("" name),(func)};
#define BENZINA_TOOL_REGISTER(name, func)                         \
    BENZINA_ATTRIBUTE_USED                                        \
    BENZINA_ATTRIBUTE_SECTION(ASM_SECT_TOOLARRAY_NAME)            \
    static const BENZ_TOOL_ENTRY __BENZINA_REGISTER(__benz_tool_entry_) = {("" name),(func)};



/**
 * @brief "Magic" variable declarations.
 * 
 * The linker script defines a number of magic variables. We declare them here
 * without defining them, to make them available to C code.
 * 
 *   - A concatenation of LICENSE text for all of the software statically
 *     linked into libbenzina.so;
 *   - An array of luaopen_*() functions to be (pre-)loaded
 *   
 * These variables are defined in magic sections of the executable assembled
 * by the linker.
 */

/*** .license ***/
BENZINA_HIDDEN extern const char license_benzina[];


/*** .lz4.cmd_array ***/
typedef struct BENZ_LZ4_ENTRY BENZ_LZ4_ENTRY;
struct BENZ_LZ4_ENTRY{
    const char* src_start, *src_end;
    char*       dst_start, *dst_end;
};
BENZINA_HIDDEN extern const BENZ_LZ4_ENTRY __lz4_cmd_array_start[];
BENZINA_HIDDEN extern const BENZ_LZ4_ENTRY __lz4_cmd_array_end[];


/*** .tool_array ***/
typedef struct BENZ_TOOL_ENTRY BENZ_TOOL_ENTRY;
struct BENZ_TOOL_ENTRY{
    const char* name; int (*func)(int argc, char* argv[]);
};
BENZINA_HIDDEN extern const BENZ_TOOL_ENTRY __tool_array_start[];
BENZINA_HIDDEN extern const BENZ_TOOL_ENTRY __tool_array_end[];


/*** .lua.open_array ***/
BENZINA_HIDDEN extern const luaL_Reg __lua_open_array_start[];
BENZINA_HIDDEN extern const luaL_Reg __lua_open_array_end[];


/*** .lua.text_array ***/
typedef struct BENZ_LUA_ENTRY BENZ_LUA_ENTRY;
struct BENZ_LUA_ENTRY{
    const char* name, *start, *end;
};
BENZINA_HIDDEN extern const BENZ_LUA_ENTRY __lua_text_array_start[];
BENZINA_HIDDEN extern const BENZ_LUA_ENTRY __lua_text_array_end[];


/* End Include Guard */
#endif

