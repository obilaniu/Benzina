/* Include Guards */
#ifndef INCLUDE_BENZINA_VISIBILITY_H
#define INCLUDE_BENZINA_VISIBILITY_H


/**
 * Includes
 */

#include "benzina/attribute.h"


/**
 * When building a library, it is a good idea to expose as few as possible
 * internal symbols (functions, objects, data structures). Not only does it
 * prevent users from relying on private portions of the library that are
 * subject to change without any notice, but it can have performance
 * advantages:
 * 
 *   - It can make shared libraries link faster at dynamic-load time.
 *   - It can make internal function calls faster by bypassing the PLT.
 * 
 * Thus, the compilation will by default hide all symbols, while the API
 * headers will explicitly mark public the few symbols the users are permitted
 * to use with a tag BENZINA_PUBLIC. We also define a tag BENZINA_HIDDEN, since
 * it may be required to explicitly tag certain C++ types as visible in order
 * for exceptions to function correctly.
 * 
 * Additional complexity comes from non-POSIX-compliant systems, which
 * artificially impose a requirement on knowing whether we are building or
 * using a DLL.
 * 
 * The above commentary and below code is inspired from
 *                   'https://gcc.gnu.org/wiki/Visibility'
 */


/**
 * Benzina library attributes (libbenzina.so)
 */

#if BENZINA_IS_SHARED
# if BENZINA_IS_BUILDING
#  define BENZINA_PUBLIC BENZINA_ATTRIBUTE_EXPORT
# else
#  define BENZINA_PUBLIC BENZINA_ATTRIBUTE_IMPORT
# endif
# define  BENZINA_HIDDEN BENZINA_ATTRIBUTE_HIDDEN
#else
# define  BENZINA_PUBLIC
# define  BENZINA_HIDDEN
#endif
#define   BENZINA_STATIC static


/**
 * Benzina plugin attributes (libbenzina-plugin-*.so)
 * 
 * As a matter of fact, plugins are always dynamically-built and -loaded, so
 * BENZINA_PLUGIN_IS_SHARED should always be #define'd.
 */

#if BENZINA_PLUGIN_IS_SHARED
# if BENZINA_PLUGIN_IS_BUILDING
#  define BENZINA_PLUGIN_PUBLIC BENZINA_ATTRIBUTE_EXPORT
# else
#  define BENZINA_PLUGIN_PUBLIC BENZINA_ATTRIBUTE_IMPORT
# endif
# define  BENZINA_PLUGIN_HIDDEN BENZINA_ATTRIBUTE_HIDDEN
#else
# define  BENZINA_PLUGIN_PUBLIC
# define  BENZINA_PLUGIN_HIDDEN
#endif
#define   BENZINA_PLUGIN_STATIC static


/* End Include Guards */
#endif
