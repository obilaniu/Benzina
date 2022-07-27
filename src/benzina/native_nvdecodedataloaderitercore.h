/* Include Guard */
#ifndef SRC_BENZINA_NVDECODEDATALOADERITERCORE_H
#define SRC_BENZINA_NVDECODEDATALOADERITERCORE_H

/**
 * Includes
 */

#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include <stddef.h>
#include <stdint.h>
#include "benzina/benzina-old.h"
#include "benzina/plugins/nvdecode.h"
#include "./native_datasetcore.h"



/* Type Definitions and Forward Declarations */
typedef struct {
	PyObject_HEAD
	void* pluginHandle;
	BENZINA_PLUGIN_NVDECODE_VTABLE* v;
	DatasetCore* datasetCore;
	PyObject* bufferObj;
	void* ctx;
} NvdecodeDataLoaderIterCore;
static PyTypeObject NvdecodeDataLoaderIterCoreType;


#endif
