/* Include Guards */
#ifndef LUABENZ_H
#define LUABENZ_H


/* Imports */
#include <stdlib.h>
#include <pthread.h>


/**
 * It is not possible to modify LUA_EXTRASPACE in luaconf.h via compiler args,
 * because it is unconditionally defined. However, this is done before the
 * #include of LUA_USER_H (which selects this header), so we may #undef it
 * here and re-#define it.
 */

#undef  LUA_EXTRASPACE
#define LUA_EXTRASPACE (2*sizeof(void*))


/**
 * The build of Lua we use is internally pthreaded, so define the internal
 * lock-macros to protect the Lua core.
 */

#define lua_lock(L)   (pthread_mutex_lock  ((pthread_mutex_t*)*(void**)lua_getextraspace((L))))
#define lua_unlock(L) (pthread_mutex_unlock((pthread_mutex_t*)*(void**)lua_getextraspace((L))))
#define luai_userstateopen(L)                                                        \
    do{                                                                              \
        lua_State* _L = (L);                                                         \
        pthread_mutex_t* _p = calloc(sizeof(pthread_mutex_t), 1);                    \
        if(!_p){                                                                     \
            lua_pushliteral(_L, MEMERRMSG);                                          \
            lua_error(L);                                                            \
        }                                                                            \
                                                                                     \
        if(pthread_mutex_init(_p, NULL) != 0){                                       \
            free(_p);                                                                \
            lua_pushliteral(_L, "Failed to initialize per-lua_State global lock!");  \
            lua_error(L);                                                            \
        }                                                                            \
                                                                                     \
        *(void**)lua_getextraspace(_L) = (void*)_p;                                  \
    }while(0)
#define luai_userstateclose(L)                                                       \
    do{                                                                              \
        pthread_mutex_t* _p = *(void**)lua_getextraspace((L));                       \
        pthread_mutex_unlock(_p); /* luai_userstateclose(L) called w/ lock */        \
        pthread_mutex_destroy(_p);                                                   \
        free(_p);                                                                    \
    }while(0)


#endif
