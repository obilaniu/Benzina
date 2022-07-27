/* Include Guard */
#ifndef SRC_BENZINA_NVDECODEDATALOADERITERCOREBATCHCM_H
#define SRC_BENZINA_NVDECODEDATALOADERITERCOREBATCHCM_H

/**
 * Includes
 */

#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include <stddef.h>
#include <stdint.h>
#include "benzina/benzina-old.h"
#include "./native_nvdecodedataloaderitercore.h"



/* Type Definitions and Forward Declarations */
typedef struct {
	PyObject_HEAD
	NvdecodeDataLoaderIterCore* core;
	PyObject*                   token;
} NvdecodeDataLoaderIterCoreBatchCM;
static PyTypeObject NvdecodeDataLoaderIterCoreBatchCMType;


#endif
