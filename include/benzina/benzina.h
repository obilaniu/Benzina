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

BENZINA_PUBLIC int          benz_init                 (void);



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

