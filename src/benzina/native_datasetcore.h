/* Include Guard */
#ifndef SRC_BENZINA_DATASETCORE_H
#define SRC_BENZINA_DATASETCORE_H

/**
 * Includes
 */

#define  PY_SSIZE_T_CLEAN     /* So we get Py_ssize_t args. */
#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include <stddef.h>
#include <stdint.h>
#include "benzina/benzina.h"



/* Type Definitions and Forward Declarations */
typedef struct{
    PyObject_HEAD
    BENZINA_DATASET* dataset;
} DatasetCore;
static PyTypeObject DatasetCoreType;


#endif
