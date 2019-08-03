/* Include Guard */
#ifndef INCLUDE_BENZINA_VERSION_H
#define INCLUDE_BENZINA_VERSION_H


/**
 * Defines
 */

#define LIBBENZINA_VERSION_MAJOR @LIBBENZINA_VERSION_MAJOR@
#define LIBBENZINA_VERSION_MINOR @LIBBENZINA_VERSION_MINOR@
#define LIBBENZINA_VERSION_PATCH @LIBBENZINA_VERSION_PATCH@

#undef  __BENZ_QUOTEME
#define __BENZ_QUOTEME(x) #x
#define LIBBENZINA_VERSION_INT(major, minor, patch) \
    ((unsigned)(major)<<16 | (unsigned)(minor)<<8 | (unsigned)(patch)<<0)
#define LIBBENZINA_VERSION(major, minor, patch) \
    __BENZ_QUOTEME(major) "." __BENZ_QUOTEME(minor) "." __BENZ_QUOTEME(patch)

#define BENZ_VERSION_MAJOR(x) (((x) | 0x00U) >> 16)
#define BENZ_VERSION_MINOR(x) (((x) & 0xFF00U) >> 8)
#define BENZ_VERSION_PATCH(x) ((x) & 0xFFU)


/* End Include Guard */
#endif

