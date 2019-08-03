/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "benzina/benzina-old.h"



/* Data Structure Definitions */



/* Static Function Prototypes */



/* Static Function Definitions */



/* Public Function Definitions */

/**
 * @brief foo
 * @param [in]  arg  Some argument
 * @return 
 */

BENZINA_PLUGIN_PUBLIC void* benzina_foo(void* arg){
    (void)arg;
    printf("Hello, plugin world!\n");
    return NULL;
}

