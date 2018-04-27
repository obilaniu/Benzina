/* Include Guard */
#ifndef SRC_BENZINA_NATIVE_H
#define SRC_BENZINA_NATIVE_H


/**
 * Includes
 */

#define  PY_SSIZE_T_CLEAN     /* So we get Py_ssize_t args. */
#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include "benzina/benzina.h"



/* Defines */




/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structure Definitions */

/**
 * @brief Python BenzinaDatasetCore object.
 */

typedef struct{
    PyObject_HEAD
    BENZINA_DATASET* dset;
} BenzinaDatasetCore;

/**
 * @brief Python BenzinaDataLoaderIterCore object.
 */

typedef struct {
	PyObject_HEAD
	BENZINA_DATALOADER_ITER* iter;
} BenzinaDataLoaderIterCore;



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

