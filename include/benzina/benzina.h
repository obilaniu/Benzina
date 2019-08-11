/* Include Guard */
#ifndef INCLUDE_BENZINA_BENZINA_H
#define INCLUDE_BENZINA_BENZINA_H


/**
 * Includes
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "benzina/bits.h"
#include "benzina/config.h"
#include "benzina/endian.h"
#include "benzina/iso.h"
#include "benzina/version.h"
#include "benzina/visibility.h"



/* Defines */



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Function Prototypes */

/**
 * @brief Benzina Initialization.
 * 
 * @return Zero if successful; Non-zero if not successful.
 */

BENZINA_PUBLIC int benz_init(void);

/**
 * @brief Get Benzina version as integer or directly access as string.
 * 
 * @return LIBBENZINA_VERSION_INT(LIBBENZINA_VERSION_MAJOR,
 *                                LIBBENZINA_VERSION_MINOR,
 *                                LIBBENZINA_VERSION_PATCH) or
 *         LIBBENZINA_VERSION    (LIBBENZINA_VERSION_MAJOR,
 *                                LIBBENZINA_VERSION_MINOR,
 *                                LIBBENZINA_VERSION_PATCH),
 *         as compiled at build time.
 */

BENZINA_PUBLIC unsigned              benz_version(void);
BENZINA_PUBLIC extern const unsigned benz_version_major;
BENZINA_PUBLIC extern const unsigned benz_version_minor;
BENZINA_PUBLIC extern const unsigned benz_version_patch;
BENZINA_PUBLIC extern const char     benz_version_str[];


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

