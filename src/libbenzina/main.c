/* Includes */
#include <stdio.h>
#include <string.h>
#include "internal.h"
#include "benzina/benzina.h"


extern int main_luac(int argc, char* argv[]);
extern int main_lua (int argc, char* argv[]);


BENZINA_STATIC char* strbasename(char* s){
    char* r = strrchr(s, '/');
    return r ? r+1 : s;
}
BENZINA_STATIC int strequals(const char* s, const char* p){
    return strcmp(s, p) == 0;
}
BENZINA_STATIC int strstartswith(const char* s, const char* prefix){
    size_t ls=strlen(s), lp=strlen(prefix);
    return ls<lp ? 0 : memcmp(s, prefix, lp)==0;
}
BENZINA_STATIC int toolequals(const char* s, const char* tool){
    return strstartswith(s, "benz-") && strequals(s+5, tool);
}


BENZINA_STATIC int main_license(int argc, char* argv[]){
    (void)argc;
    (void)argv;
    printf("%s\n", license_benzina);
    return 0;
}
BENZINA_STATIC int main_none(int argc, char* argv[]){
    (void)argc;
    (void)argv;
    printf("Benzina version: %s\n"
           "Configuration:\n",
           LIBBENZINA_VERSION_STR(LIBBENZINA_VERSION_MAJOR,
                                  LIBBENZINA_VERSION_MINOR,
                                  LIBBENZINA_VERSION_PATCH));
    return 0;
}


/**
 * @brief Entry Point of libbenzina.so
 * 
 * The shared library libbenzina.so may be executed as if it were a command. In
 * this respect it is somewhat akin to libc.so.6 and /lib64/ld-linux-x86-64.so.2.
 * 
 * To decide what to do, libbenzina.so keys off the name it was invoked as (argv[0]),
 * plus any subcommand arguments it was provided. In this respect it is akin to
 * the Busybox distribution.
 * 
 * When...
 *   1. Basename *begins with* "luac", act as a Lua compiler.
 *   2. Basename *begins with* "lua",  act as a Lua interpreter.
 *   3. Otherwise:
 *     3.1. If invoked with no arguments, print version and configuration.
 *     3.2. Otherwise, inspect the first argument, a subcommand:
 *        3.2.1. If invoked with argv[1] *equal* to "luac", act as a Lua compiler.
 *        3.2.2. If invoked with argv[1] *equal* to "lua",  act as a Lua interpreter.
 *        3.2.3. If invoked with argv[1] *equal* to "license" or "licenses", print licenses.
 * 
 * @param [in] argc  The classic main() argument count  mandated by Standard C.
 * @param [in] argv  The classic main() argument vector mandated by Standard C.
 * @return Exit code.
 */

BENZINA_HIDDEN int main(int argc, char* argv[]){
    char* basename;
    const BENZ_TOOL_ENTRY* t;
    
    
    /* Should we be execve()'d with no args or (more strangely) no name, simply
     * print our own configuration. */
    char  fake_arg0[] = "";
    char* fake_argv[] = {fake_arg0, NULL};
    if(argc<=0 || !argv[0] || !*argv[0])
        return main_none(1, fake_argv);
    
    
    /* Otherwise, check if the basename is empty, prefixed with luac/lua or
     * the single argument and dispatch these as special cases. */
    basename = strbasename(argv[0]);
    if(!basename || !*basename)              return main_none(argc, argv);
    else if(strstartswith(basename, "luac")) return main_luac(argc, argv);
    else if(strstartswith(basename, "lua"))  return main_lua (argc, argv);
    
    
    /* Dispatch on tool match in basename of first argument:
     * `benz-<toolname>` */
    for(t=__tool_array_start; t<__tool_array_end; t++)
        if(toolequals(basename, t->name))
            return t->func(argc, argv);
    
    
    /* If called with any other name and only one argument, print own
     * configuration. */
    if(argc <= 1) return main_none(argc, argv);
    
    
    /* Dispatch on exact match in second argument: */
    for(t=__tool_array_start; t<__tool_array_end; t++)
        if(strequals(argv[1], t->name))
            return t->func(argc-1, argv+1);
    
    
    /* Unknown tool. */
    return main_none(argc, argv);
}


/* Register tools */
BENZINA_TOOL_REGISTER("",         main_none)
BENZINA_TOOL_REGISTER("lua",      main_lua)
BENZINA_TOOL_REGISTER("luac",     main_luac)
BENZINA_TOOL_REGISTER("license",  main_license)
BENZINA_TOOL_REGISTER("licenses", main_license)
