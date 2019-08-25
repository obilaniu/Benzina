/* Includes */
#include "benzina/benzina.h"



/* Public Function Definitions */

/**
 * Get library version.
 */

BENZINA_PUBLIC uint32_t       benz_version(void){
    return LIBBENZINA_VERSION_INT(LIBBENZINA_VERSION_MAJOR,
                                  LIBBENZINA_VERSION_MINOR,
                                  LIBBENZINA_VERSION_PATCH);
}
BENZINA_PUBLIC const uint32_t benz_version_major = LIBBENZINA_VERSION_MAJOR;
BENZINA_PUBLIC const uint32_t benz_version_minor = LIBBENZINA_VERSION_MINOR;
BENZINA_PUBLIC const uint32_t benz_version_patch = LIBBENZINA_VERSION_PATCH;
BENZINA_PUBLIC const uint32_t benz_version_int   =
        LIBBENZINA_VERSION_INT(
            LIBBENZINA_VERSION_MAJOR,
            LIBBENZINA_VERSION_MINOR,
            LIBBENZINA_VERSION_PATCH
        );
BENZINA_PUBLIC const char     benz_version_str[] =
        LIBBENZINA_VERSION_STR(
            LIBBENZINA_VERSION_MAJOR,
            LIBBENZINA_VERSION_MINOR,
            LIBBENZINA_VERSION_PATCH
        );
