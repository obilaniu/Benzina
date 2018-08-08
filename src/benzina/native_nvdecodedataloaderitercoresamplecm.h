/* Include Guard */
#ifndef SRC_BENZINA_NVDECODEDATALOADERITERCORESAMPLECM_H
#define SRC_BENZINA_NVDECODEDATALOADERITERCORESAMPLECM_H

/**
 * Includes
 */

#define  PY_SSIZE_T_CLEAN     /* So we get Py_ssize_t args. */
#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include <stddef.h>
#include <stdint.h>
#include "benzina/benzina.h"
#include "./native_nvdecodedataloaderitercorebatchcm.h"



/* Type Definitions and Forward Declarations */
typedef struct {
	PyObject_HEAD
	NvdecodeDataLoaderIterCoreBatchCM* batch;
	uint64_t                           index;
	void*                              dstPtr;
} NvdecodeDataLoaderIterCoreSampleCM;
static PyTypeObject NvdecodeDataLoaderIterCoreSampleCMType;


#endif
