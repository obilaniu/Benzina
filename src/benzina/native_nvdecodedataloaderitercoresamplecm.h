/* Include Guard */
#ifndef SRC_BENZINA_NVDECODEDATALOADERITERCORESAMPLECM_H
#define SRC_BENZINA_NVDECODEDATALOADERITERCORESAMPLECM_H

/**
 * Includes
 */

#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include <stddef.h>
#include <stdint.h>
#include "benzina/benzina-old.h"
#include "./native_nvdecodedataloaderitercorebatchcm.h"



/* Type Definitions and Forward Declarations */
typedef struct {
	PyObject_HEAD
	NvdecodeDataLoaderIterCoreBatchCM* batch;
	uint64_t                           index;
	void*                              dstPtr;
	void*                              sample;
	uint64_t                           location[2];
	uint64_t                           config_location[2];
} NvdecodeDataLoaderIterCoreSampleCM;
static PyTypeObject NvdecodeDataLoaderIterCoreSampleCMType;


#endif
