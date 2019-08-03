/* Includes */
#include "benzina/benzina.h"



/* Public Function Definitions */

/**
 * Get library version.
 */

BENZINA_PUBLIC unsigned   benz_version(void){
    return LIBBENZINA_VERSION_INT(LIBBENZINA_VERSION_MAJOR,
                                  LIBBENZINA_VERSION_MINOR,
                                  LIBBENZINA_VERSION_PATCH);
}
BENZINA_PUBLIC const char benz_version_str[] =
        LIBBENZINA_VERSION(
            LIBBENZINA_VERSION_MAJOR,
            LIBBENZINA_VERSION_MINOR,
            LIBBENZINA_VERSION_PATCH
        );
