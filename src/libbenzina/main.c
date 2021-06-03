/* Includes */
#include <stdio.h>
#include <string.h>
#include "benzina/benzina.h"


extern const char linenoise_license[];
extern const char lua_license[];
extern const char lfs_license[];
extern int luac_main(int argc, char* argv[]);
extern int lua_main (int argc, char* argv[]);



BENZINA_STATIC const char* strbasename(const char* s){
    const char* r = strrchr(s, '/');
    return r ? r+1 : s;
}
BENZINA_STATIC int strequals(const char* s, const char* p){
    return strcmp(s, p) == 0;
}
BENZINA_STATIC int strstartswith(const char* s, const char* prefix){
    size_t ls=strlen(s), lp=strlen(prefix);
    return ls<lp ? 0 : memcmp(s, prefix, lp)==0;
}


BENZINA_STATIC int license_main(int argc, char* argv[]){
    (void)argc;
    (void)argv;
    printf("Printing licenses!\n%s\n%s\n%s\n",
           linenoise_license,
           lua_license,
           lfs_license);
    return 0;
}
BENZINA_STATIC int no_main(int argc, char* argv[]){
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
    const char* basename;
    
    
    /* Should we be execve()'d with no args or (more strangely) no name, simply
     * print our own configuration. */
    if(argc<=0 || !argv[0] || !*argv[0])
        return no_main(argc, argv);
    
    basename = strbasename(argv[0]);
    if(!basename)
        return no_main(argc, argv);
    
    
    if     (strstartswith(basename, "luac"))
        return luac_main(argc, argv);
    else if(strstartswith(basename, "lua"))
        return lua_main(argc, argv);
    
    
    if(argc<=1)
        return no_main(argc, argv);
    
    
    if     (strequals(argv[1], "luac"))
        return luac_main(argc-1, argv+1);
    else if(strequals(argv[1], "lua"))
        return lua_main(argc-1, argv+1);
    else if(strequals(argv[1], "license") ||
            strequals(argv[1], "licenses"))
        return license_main(argc-1, argv+1);
    
    
    return no_main(argc, argv);
}
