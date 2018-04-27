/* Includes */
#define  PY_SSIZE_T_CLEAN  /* So we get Py_ssize_t args. */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include "structmember.h"
#include <stdint.h>
#include "native.h"


/* Defines */



/* Python API Function Definitions */

/**
 * @brief Slot tp_dealloc
 */

static void      BenzinaDataLoaderIterCore_dealloc  (BenzinaDataLoaderIterCore* self){
	benzinaDataLoaderIterFree(self->iter);
	Py_TYPE(self)->tp_free(self);
}

/**
 * @brief Slot tp_new
 */

static PyObject* BenzinaDataLoaderIterCore_new      (PyTypeObject* type,
                                                     PyObject*     args,
                                                     PyObject*     kwargs){
	BenzinaDataLoaderIterCore* self = (BenzinaDataLoaderIterCore*)type->tp_alloc(type, 0);
	
	self->iter = NULL;
	
	return (PyObject*)self;
}

/**
 * @brief Slot tp_init
 */

static int       BenzinaDataLoaderIterCore_init     (BenzinaDataLoaderIterCore* self,
                                                     PyObject*                  args,
                                                     PyObject*                  kwargs){
	BenzinaDatasetCore* dset=NULL;
	int                 deviceId;
	Py_ssize_t          multibuffering;
	Py_ssize_t          batchSize;
	Py_ssize_t          h, w;
	
	static char *kwargsList[] = {"dataset",
	                             "deviceId",
	                             "multibuffering",
	                             "batchSize",
	                             "h",
	                             "w",
	                             NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Oinnnn", kwargsList, &dset,
	                                &deviceId, &multibuffering, &batchSize, &h, &w)){
		return -1;
	}
	
	if(benzinaDataLoaderIterNew(&self->iter,
	                            dset->dset,
	                            deviceId,
	                            multibuffering,
	                            batchSize,
	                            h, w) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error during creation of underlying "
		                "BENZINA_DATALOADER_ITER* object.");
		Py_DECREF(dset);
		return -1;
	}
	
	return 0;
}

static PyObject*  BenzinaDataLoaderIterCore_getfirst(BenzinaDataLoaderIterCore *self, void *closure){
	return NULL;
}

static PyObject*  BenzinaDataLoaderIterCore_getlast(BenzinaDataLoaderIterCore* self,
                                                    void*                      closure){
	return NULL;
}

static PyGetSetDef BenzinaDataLoaderIterCore_getsetters[] = {
	{"first", (getter)BenzinaDataLoaderIterCore_getfirst, 0, "first batch", NULL},
	{"last",  (getter)BenzinaDataLoaderIterCore_getlast,  0, "last batch",  NULL},
	{NULL}  /* Sentinel */
};

static PyObject * BenzinaDataLoaderIterCore_push(BenzinaDataLoaderIterCore* self,
                                                 PyObject*                  args,
                                                 PyObject*                  kwargs){
	return NULL;
}

static PyObject * BenzinaDataLoaderIterCore_pull(BenzinaDataLoaderIterCore* self){
	return NULL;
}

static PyMethodDef BenzinaDataLoaderIterCore_methods[] = {
    {"push", (PyCFunction)BenzinaDataLoaderIterCore_push, METH_VARARGS|METH_KEYWORDS, "Push a new batch's worth of work."},
    {"pull", (PyCFunction)BenzinaDataLoaderIterCore_pull, METH_NOARGS,                "Pull an old batch's worth of work."},
	{NULL}  /* Sentinel */
};

static PyTypeObject BenzinaDataLoaderIterCoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "native.BenzinaDataLoaderIterCore",             /* tp_name */
    sizeof(BenzinaDataLoaderIterCore),              /* tp_basicsize */
    0,                                              /* tp_itemsize */
    (destructor)BenzinaDataLoaderIterCore_dealloc,  /* tp_dealloc */
    0,                                              /* tp_print */
    0,                                              /* tp_getattr */
    0,                                              /* tp_setattr */
    0,                                              /* tp_reserved */
    0,                                              /* tp_repr */
    0,                                              /* tp_as_number */
    0,                                              /* tp_as_sequence */
    0,                                              /* tp_as_mapping */
    0,                                              /* tp_hash  */
    0,                                              /* tp_call */
    0,                                              /* tp_str */
    0,                                              /* tp_getattro */
    0,                                              /* tp_setattro */
    0,                                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,         /* tp_flags */
    "BenzinaDataLoaderIterCore objects",            /* tp_doc */
    0,                                              /* tp_traverse */
    0,                                              /* tp_clear */
    0,                                              /* tp_richcompare */
    0,                                              /* tp_weaklistoffset */
    0,                                              /* tp_iter */
    0,                                              /* tp_iternext */
    BenzinaDataLoaderIterCore_methods,              /* tp_methods */
    0,                                              /* tp_members */
    BenzinaDataLoaderIterCore_getsetters,           /* tp_getset */
    0,                                              /* tp_base */
    0,                                              /* tp_dict */
    0,                                              /* tp_descr_get */
    0,                                              /* tp_descr_set */
    0,                                              /* tp_dictoffset */
    (initproc)BenzinaDataLoaderIterCore_init,       /* tp_init */
    0,                                              /* tp_alloc */
    BenzinaDataLoaderIterCore_new,                  /* tp_new */
};
