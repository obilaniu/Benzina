/* Includes */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include <stdint.h>
#include "./native_nvdecodedataloaderitercoresamplecm.h"



/* Python API Function Definitions */

/**
 * @brief Slot tp_new
 */

static PyObject* NvdecodeDataLoaderIterCoreSampleCM_new             (PyTypeObject* type,
                                                                     PyObject*     args,
                                                                     PyObject*     kwargs){
	(void)args;
	(void)kwargs;
	
	NvdecodeDataLoaderIterCoreSampleCM* self = (NvdecodeDataLoaderIterCoreSampleCM*)type->tp_alloc(type, 0);
	
	self->batch  = NULL;
	self->index  = 0;
	self->dstPtr = NULL;
	self->sample = NULL;
	self->location[0] = self->location[1] = 0;
	self->config_location[0] = self->config_location[1] = 0;
	
	return (PyObject*)self;
}

/**
 * @brief Slot tp_dealloc
 */

static void      NvdecodeDataLoaderIterCoreSampleCM_dealloc         (NvdecodeDataLoaderIterCoreSampleCM* self){
	//PyObject_GC_UnTrack(self);
	Py_TYPE(self)->tp_clear((PyObject*)self);
	Py_TYPE(self)->tp_free(self);
}

/**
 * @brief Slot tp_traverse
 */

static int       NvdecodeDataLoaderIterCoreSampleCM_traverse        (NvdecodeDataLoaderIterCoreSampleCM* self,
                                                                     visitproc                           visit,
                                                                     void*                               arg){
	Py_VISIT(self->batch);
	return 0;
}

/**
 * @brief Slot tp_clear
 */

static int       NvdecodeDataLoaderIterCoreSampleCM_clear           (NvdecodeDataLoaderIterCoreSampleCM* self){
	Py_CLEAR(self->batch);
	return 0;
}

/**
 * @brief Slot tp_init
 */

static int       NvdecodeDataLoaderIterCoreSampleCM_init            (NvdecodeDataLoaderIterCoreSampleCM* self,
                                                                     PyObject*                           args,
                                                                     PyObject*                           kwargs){
	NvdecodeDataLoaderIterCoreBatchCM* batch               = NULL;
	unsigned long long                 index               = 0;
	unsigned long long                 dstPtr              = 0;
	PyObject*                          sample              = NULL;
	PyObject*                          location            = NULL;
	PyObject*                          config_location     = NULL;
	
	static char *kwargsList[] = {"batch", "index", "dstPtr", "sample", "location", "config_location", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OKKOOO", kwargsList,
	                                &batch, &index, &dstPtr, &sample, &location, &config_location)){
		return -1;
	}
	
	if(!PyMemoryView_Check(sample)){
		Py_DECREF(sample);
		sample = NULL;
	}
	
	if(!PyTuple_Check(location) ||
	   !PyArg_ParseTuple(location, "kk", &self->location[0], &self->location[1])){
        Py_DECREF(location);
        location = NULL;
	}

	if(!PyTuple_Check(config_location) ||
	   !PyArg_ParseTuple(config_location, "kk", &self->config_location[0], &self->config_location[1])){
        Py_DECREF(config_location);
        config_location = NULL;
	}

	if(sample == NULL || location == NULL || config_location == NULL){
        return -1;
    }
	
	Py_INCREF(batch);
	self->batch  = batch;
	self->index  = index;
	self->dstPtr = (void*)dstPtr;
	self->sample = PyMemoryView_GET_BUFFER(sample)->buf;
	// Ownership of sample is kept in benzina.torch.dataloader._DataLoaderIter
	// Py_DECREF(sample);
	Py_DECREF(location);
	Py_DECREF(config_location);
	//PyObject_GC_Track(self);
	return 0;
}


/**
 * METHODS
 */

static PyObject* NvdecodeDataLoaderIterCoreSampleCM___enter__       (NvdecodeDataLoaderIterCoreSampleCM* self){
	if(self->batch->core->v->defineSample(self->batch->core->ctx,
	                                      self->index,
	                                      self->dstPtr,
	                                      self->sample,
	                                      self->location,
	                                      self->config_location) != 0){
		PyErr_SetString(PyExc_RuntimeError, "Error attempting to define a sample!");
		return NULL;
	}else{
		Py_INCREF(self);
		return (PyObject*)self;
	}
}
static PyObject* NvdecodeDataLoaderIterCoreSampleCM___exit__        (NvdecodeDataLoaderIterCoreSampleCM* self,
                                                                     PyObject*                           args,
                                                                     PyObject*                           kwargs){
	static char *kwargsList[] = {"type", "value", "traceback", NULL};
	
	PyObject* type=NULL, *value=NULL, *traceback=NULL;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwargsList,
	                                &type, &value, &traceback)){
		return NULL;
	}
	
	if(type==Py_None && value==Py_None && traceback==Py_None){
		/**
		 * No exceptions between __enter__() and __exit__().
		 * Attempt sample submit.
		 */
		
		if(self->batch->core->v->submitSample(self->batch->core->ctx) != 0){
			PyErr_SetString(PyExc_RuntimeError, "Error attempting to submit a sample!");
			return NULL;
		}else{
			Py_INCREF(Py_None);
			return Py_None;
		}
	}else{
		/**
		 * An exception was raised somewhere between __enter__() and __exit__().
		 * Difficult to decide what to do; We decide not to submit the sample and
		 * not to swallow the exception. We do this by returning a False-y value.
		 */
		
		Py_INCREF(Py_None);
		return Py_None;
	}
}


/**
 * Methods table.
 */

static PyMethodDef NvdecodeDataLoaderIterCoreSampleCM_methods[] = {
    {"__enter__", (PyCFunction)NvdecodeDataLoaderIterCoreSampleCM___enter__, METH_NOARGS,                "Enter a sample definition context."},
    {"__exit__",  (PyCFunction)NvdecodeDataLoaderIterCoreSampleCM___exit__,  METH_VARARGS|METH_KEYWORDS, "Exit the sample definition context and submit it."},
    {NULL}  /* Sentinel */
};

static PyTypeObject NvdecodeDataLoaderIterCoreSampleCMType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "benzina.native.NvdecodeDataLoaderIterCoreSampleCM",       /* tp_name */
    sizeof(NvdecodeDataLoaderIterCoreSampleCM),                /* tp_basicsize */
    0,                                                         /* tp_itemsize */
    (destructor)NvdecodeDataLoaderIterCoreSampleCM_dealloc,    /* tp_dealloc */
    0,                                                         /* tp_print */
    0,                                                         /* tp_getattr */
    0,                                                         /* tp_setattr */
    0,                                                         /* tp_reserved */
    0,                                                         /* tp_repr */
    0,                                                         /* tp_as_number */
    0,                                                         /* tp_as_sequence */
    0,                                                         /* tp_as_mapping */
    0,                                                         /* tp_hash  */
    0,                                                         /* tp_call */
    0,                                                         /* tp_str */
    0,                                                         /* tp_getattro */
    0,                                                         /* tp_setattro */
    0,                                                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,                   /* tp_flags */
    "NvdecodeDataLoaderIterCoreSampleCM object",               /* tp_doc */
    (traverseproc)NvdecodeDataLoaderIterCoreSampleCM_traverse, /* tp_traverse */
    (inquiry)NvdecodeDataLoaderIterCoreSampleCM_clear,         /* tp_clear */
    0,                                                         /* tp_richcompare */
    0,                                                         /* tp_weaklistoffset */
    0,                                                         /* tp_iter */
    0,                                                         /* tp_iternext */
    NvdecodeDataLoaderIterCoreSampleCM_methods,                /* tp_methods */
    0,                                                         /* tp_members */
    0,                                                         /* tp_getset */
    0,                                                         /* tp_base */
    0,                                                         /* tp_dict */
    0,                                                         /* tp_descr_get */
    0,                                                         /* tp_descr_set */
    0,                                                         /* tp_dictoffset */
    (initproc)NvdecodeDataLoaderIterCoreSampleCM_init,         /* tp_init */
    0,                                                         /* tp_alloc */
    (newfunc)NvdecodeDataLoaderIterCoreSampleCM_new,           /* tp_new */
    0,                                                         /* tp_free */
    0,                                                         /* tp_is_gc */
    0,                                                         /* tp_bases */
    0,                                                         /* tp_mro */
    0,                                                         /* tp_cache */
    0,                                                         /* tp_subclasses */
    0,                                                         /* tp_weaklist */
    0,                                                         /* tp_del */
    0,                                                         /* tp_version_tag */
    0,                                                         /* tp_finalize */
};
