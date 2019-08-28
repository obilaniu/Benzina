/* Includes */
#include "benzina/benzina.h"



/* Public Function Definitions */
BENZINA_PUBLIC uint32_t       benz_version(void){
    return LIBBENZINA_VERSION_INT(LIBBENZINA_VERSION_MAJOR,
                                  LIBBENZINA_VERSION_MINOR,
                                  LIBBENZINA_VERSION_PATCH);
}
BENZINA_PUBLIC const char*    benz_license(void){
    return "MIT";
}
BENZINA_PUBLIC const char*    benz_configuration(void){
    return "";
}
