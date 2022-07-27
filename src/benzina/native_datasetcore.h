/* Include Guard */
#ifndef SRC_BENZINA_DATASETCORE_H
#define SRC_BENZINA_DATASETCORE_H

/**
 * Includes
 */

#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include <stddef.h>
#include <stdint.h>
#include "benzina/benzina-old.h"



/* Type Definitions and Forward Declarations */
typedef struct{
    PyObject_HEAD
    BENZINA_DATASET* dataset;
} DatasetCore;
static PyTypeObject DatasetCoreType;


#endif
