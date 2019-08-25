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
    ((uint32_t)(major)<<16 | (uint32_t)(minor)<<8 | (uint32_t)(patch)<<0)
#define LIBBENZINA_VERSION_STR(major, minor, patch) \
    __BENZ_QUOTEME(major) "." __BENZ_QUOTEME(minor) "." __BENZ_QUOTEME(patch)

#define BENZ_VERSION_MAJOR(x) (((uint32_t)(x)) >> 16)
#define BENZ_VERSION_MINOR(x) (((uint32_t)(x) & 0xFF00U) >> 8)
#define BENZ_VERSION_PATCH(x) ((uint32_t)(x) & 0xFFU)


/* End Include Guard */
#endif

