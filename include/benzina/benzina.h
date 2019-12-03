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
#include "benzina/intops.h"
#include "benzina/iso/iso.h"
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

BENZINA_PUBLIC int                   benz_init(void);

/**
 * @brief Get Benzina version as integer.
 * 
 * @return LIBBENZINA_VERSION_INT(LIBBENZINA_VERSION_MAJOR,
 *                                LIBBENZINA_VERSION_MINOR,
 *                                LIBBENZINA_VERSION_PATCH),
 *         as compiled at build time.
 */

BENZINA_PUBLIC uint32_t              benz_version(void);

/**
 * @brief Get Benzina license as a string.
 */

BENZINA_PUBLIC extern const char*    benz_license(void);

/**
 * @brief Get Benzina configuration as a string.
 */

BENZINA_PUBLIC extern const char*    benz_configuration(void);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

