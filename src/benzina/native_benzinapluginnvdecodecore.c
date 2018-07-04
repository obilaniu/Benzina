/* Includes */
#define  PY_SSIZE_T_CLEAN  /* So we get Py_ssize_t args. */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include "structmember.h"
#include <dlfcn.h>
#include <stdint.h>
#include "native.h"


/* Defines */



/* Python API Function Definitions */

/**
 * @brief Slot tp_dealloc
 */

static void      BenzinaPluginNvdecodeCore_dealloc  (BenzinaPluginNvdecodeCore* self){
	self->v->release(self->ctx);
	dlclose(self->pluginHandle);
	Py_XDECREF(self->datasetCore);
	self->datasetCore  = NULL;
	self->pluginHandle = NULL;
	self->v            = NULL;
	self->ctx          = NULL;
	Py_TYPE(self)->tp_free(self);
}

/**
 * @brief Slot tp_new
 */

static PyObject* BenzinaPluginNvdecodeCore_new      (PyTypeObject* type,
                                                     PyObject*     args,
                                                     PyObject*     kwargs){
	void* pluginHandle, *v;
	
	pluginHandle = dlopen("libbenzina_plugin_nvdecode.so", RTLD_LAZY);
	if(!pluginHandle){
		PyErr_SetString(PyExc_ImportError, "Could not load libbenzina_plugin_nvdecode.so!");
		return NULL;
	}
	v = dlsym(pluginHandle, "VTABLE");
	if(!v){
		dlclose(pluginHandle);
		PyErr_SetString(PyExc_ImportError, "Incompatible libbenzina_plugin_nvdecode.so found!");
		return NULL;
	}
	
	BenzinaPluginNvdecodeCore* self = (BenzinaPluginNvdecodeCore*)type->tp_alloc(type, 0);
	
	self->pluginHandle = pluginHandle;
	self->v            = v;
	self->datasetCore  = NULL;
	self->ctx          = NULL;
	
	return (PyObject*)self;
}

/**
 * @brief Slot tp_init
 */

static int       BenzinaPluginNvdecodeCore_init     (BenzinaPluginNvdecodeCore* self,
                                                     PyObject*                  args,
                                                     PyObject*                  kwargs){
	BenzinaDatasetCore* datasetCore    = NULL;
	const char*         deviceId       = "cuda:0";
	unsigned long long  devicePtr      = 0;
	unsigned long long  batchSize      = 256;
	unsigned long long  multibuffering = 3;
	unsigned long long  outputHeight   = 256;
	unsigned long long  outputWidth    = 256;
	
	static char *kwargsList[] = {"dataset",
	                             "deviceId",
	                             "devicePtr",
	                             "batchSize",
	                             "multibuffering",
	                             "outputHeight",
	                             "outputWidth",
	                             NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OsK|KKKKK", kwargsList,
	                                &datasetCore,  &deviceId,    &devicePtr,
	                                &batchSize,    &multibuffering,
	                                &outputHeight, &outputWidth)){
		return -1;
	}
	
	if(self->v->alloc(&self->ctx, datasetCore->dataset) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error during creation of decoding context!");
		return -1;
	}
	
	if(self->v->setBuffer(self->ctx, deviceId, (void*)devicePtr, multibuffering,
	                      batchSize, outputHeight, outputWidth) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error during installation of decoding context's target"
		                " buffer and geometry!");
		return -1;
	}
	
	if(self->v->init(self->ctx) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Failed initializing decoder context!");
		return -1;
	}
	
	self->datasetCore = datasetCore;
	Py_INCREF(self->datasetCore);
	return 0;
}

static PyObject*  BenzinaPluginNvdecodeCore_getfirst(BenzinaPluginNvdecodeCore *self, void *closure){
	return NULL;
}

static PyObject*  BenzinaPluginNvdecodeCore_getlast(BenzinaPluginNvdecodeCore* self,
                                                    void*                      closure){
	return NULL;
}

static PyGetSetDef BenzinaPluginNvdecodeCore_getsetters[] = {
	{"first", (getter)BenzinaPluginNvdecodeCore_getfirst, 0, "first batch", NULL},
	{"last",  (getter)BenzinaPluginNvdecodeCore_getlast,  0, "last batch",  NULL},
	{NULL}  /* Sentinel */
};

static PyObject * BenzinaPluginNvdecodeCore_push(BenzinaPluginNvdecodeCore* self,
                                                 PyObject*                  args,
                                                 PyObject*                  kwargs){
	return NULL;
}

static PyObject * BenzinaPluginNvdecodeCore_pull(BenzinaPluginNvdecodeCore* self){
	return NULL;
}

static PyMethodDef BenzinaPluginNvdecodeCore_methods[] = {
    {"push", (PyCFunction)BenzinaPluginNvdecodeCore_push, METH_VARARGS|METH_KEYWORDS, "Push a new batch's worth of work."},
    {"pull", (PyCFunction)BenzinaPluginNvdecodeCore_pull, METH_NOARGS,                "Pull an old batch's worth of work."},
	{NULL}  /* Sentinel */
};

static PyTypeObject BenzinaPluginNvdecodeCoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "native.BenzinaPluginNvdecodeCore",             /* tp_name */
    sizeof(BenzinaPluginNvdecodeCore),              /* tp_basicsize */
    0,                                              /* tp_itemsize */
    (destructor)BenzinaPluginNvdecodeCore_dealloc,  /* tp_dealloc */
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
    "BenzinaPluginNvdecodeCore objects",            /* tp_doc */
    0,                                              /* tp_traverse */
    0,                                              /* tp_clear */
    0,                                              /* tp_richcompare */
    0,                                              /* tp_weaklistoffset */
    0,                                              /* tp_iter */
    0,                                              /* tp_iternext */
    BenzinaPluginNvdecodeCore_methods,              /* tp_methods */
    0,                                              /* tp_members */
    BenzinaPluginNvdecodeCore_getsetters,           /* tp_getset */
    0,                                              /* tp_base */
    0,                                              /* tp_dict */
    0,                                              /* tp_descr_get */
    0,                                              /* tp_descr_set */
    0,                                              /* tp_dictoffset */
    (initproc)BenzinaPluginNvdecodeCore_init,       /* tp_init */
    0,                                              /* tp_alloc */
    BenzinaPluginNvdecodeCore_new,                  /* tp_new */
};
