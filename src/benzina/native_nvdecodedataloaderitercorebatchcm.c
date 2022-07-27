/* Includes */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include <stdint.h>
#include "./native_nvdecodedataloaderitercorebatchcm.h"
#include "./native_nvdecodedataloaderitercoresamplecm.h"



/* Python API Function Definitions */

/**
 * @brief Slot tp_new
 */

static PyObject* NvdecodeDataLoaderIterCoreBatchCM_new              (PyTypeObject* type,
                                                                     PyObject*     args,
                                                                     PyObject*     kwargs){
	(void)args;
	(void)kwargs;
	
	NvdecodeDataLoaderIterCoreBatchCM* self = (NvdecodeDataLoaderIterCoreBatchCM*)type->tp_alloc(type, 0);
	
	self->core  = NULL;
	self->token = NULL;
	
	return (PyObject*)self;
}

/**
 * @brief Slot tp_dealloc
 */

static void      NvdecodeDataLoaderIterCoreBatchCM_dealloc          (NvdecodeDataLoaderIterCoreBatchCM* self){
	//PyObject_GC_UnTrack(self);
	Py_TYPE(self)->tp_clear((PyObject*)self);
	Py_TYPE(self)->tp_free(self);
}

/**
 * @brief Slot tp_traverse
 */

static int       NvdecodeDataLoaderIterCoreBatchCM_traverse         (NvdecodeDataLoaderIterCoreBatchCM* self,
                                                                     visitproc                          visit,
                                                                     void*                              arg){
	Py_VISIT(self->core);
	Py_VISIT(self->token);
	return 0;
}

/**
 * @brief Slot tp_clear
 */

static int       NvdecodeDataLoaderIterCoreBatchCM_clear            (NvdecodeDataLoaderIterCoreBatchCM* self){
	Py_CLEAR(self->core);
	Py_CLEAR(self->token);
	return 0;
}

/**
 * @brief Slot tp_init
 */

static int       NvdecodeDataLoaderIterCoreBatchCM_init             (NvdecodeDataLoaderIterCoreBatchCM* self,
                                                                     PyObject*                          args,
                                                                     PyObject*                          kwargs){
	NvdecodeDataLoaderIterCore* core  = NULL;
	PyObject*                   token = NULL;
	
	static char *kwargsList[] = {"core", "token", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwargsList,
	                                &core, &token)){
		return -1;
	}
	
	if(token == NULL){
		token = Py_None;
	}
	
	Py_INCREF(core);
	Py_INCREF(token);
	Py_CLEAR (self->core);
	Py_CLEAR (self->token);
	self->core  = core;
	self->token = token;
	//PyObject_GC_Track(self);
	return 0;
}


/**
 * METHODS
 */

static PyObject* NvdecodeDataLoaderIterCoreBatchCM___enter__        (NvdecodeDataLoaderIterCoreBatchCM* self){
	if(self->core->v->defineBatch(self->core->ctx) != 0){
		PyErr_SetString(PyExc_RuntimeError, "Error attempting to define a batch!");
		return NULL;
	}else{
		Py_INCREF(self);
		return (PyObject*)self;
	}
}
static PyObject* NvdecodeDataLoaderIterCoreBatchCM___exit__         (NvdecodeDataLoaderIterCoreBatchCM* self,
                                                                     PyObject*                          args,
                                                                     PyObject*                          kwargs){
	static char *kwargsList[] = {"type", "value", "traceback", NULL};
	
	PyObject* type=NULL, *value=NULL, *traceback=NULL;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwargsList,
	                                &type, &value, &traceback)){
		return NULL;
	}
	
	if(type==Py_None && value==Py_None && traceback==Py_None){
		/**
		 * No exceptions between __enter__() and __exit__().
		 * Attempt batch submit.
		 */
		
		Py_XINCREF(self->token);
		if(self->core->v->submitBatch(self->core->ctx, self->token) != 0){
			Py_XDECREF(self->token);
			PyErr_SetString(PyExc_RuntimeError, "Error attempting to submit a batch!");
			return NULL;
		}else{
			Py_INCREF(Py_None);
			return Py_None;
		}
	}else{
		/**
		 * An exception was raised somewhere between __enter__() and __exit__().
		 * Difficult to decide what to do; We decide not to submit the batch and
		 * not to swallow the exception. We do this by returning a False-y value.
		 */
		
		Py_INCREF(Py_None);
		return Py_None;
	}
}
static PyObject* NvdecodeDataLoaderIterCoreBatchCM_setToken         (NvdecodeDataLoaderIterCoreBatchCM* self,
                                                                     PyObject*                          args,
                                                                     PyObject*                          kwargs){
	static char *kwargsList[] = {"token", NULL};
	
	PyObject* token=NULL;
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwargsList, &token)){
		return NULL;
	}
	
	if(token == NULL){
		token = Py_None;
	}
	
	Py_INCREF(token);
	Py_CLEAR (self->token);
	self->token = token;
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* NvdecodeDataLoaderIterCoreBatchCM_sample           (NvdecodeDataLoaderIterCoreBatchCM* self,
                                                                     PyObject*                          args,
                                                                     PyObject*                          kwargs){
	PyObject* ret                      = NULL;
	unsigned long long index           = 0;
	unsigned long long dstPtr          = 0;
	PyObject*          sample          = NULL;
	PyTupleObject*     location        = NULL;
	PyTupleObject*     config_location = NULL;

	static char *kwargsList[] = {"index", "dstPtr", "sample", "location", "config_location", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "KKOOO", kwargsList,
	                                &index, &dstPtr, &sample, &location, &config_location)){
		return NULL;
	}

	if(sample != NULL && !PyMemoryView_Check(sample)){
		Py_DECREF(sample);
		sample = NULL;
	}

	if(location != NULL && !PyTuple_Check(location)){
        Py_DECREF(location);
        location = NULL;
	}
	
	if(config_location != NULL && !PyTuple_Check(config_location)){
        Py_DECREF(config_location);
        config_location = NULL;
	}
	
	if(sample == NULL || location == NULL || config_location == NULL){
        return NULL;
    }
	
    ret = PyObject_CallFunction((PyObject*)&NvdecodeDataLoaderIterCoreSampleCMType,
	                            "OKKOOO", self, index, dstPtr, sample, location, config_location);
    // Py_DECREF(sample);
    Py_DECREF(location);
    Py_DECREF(config_location);
    return ret;
}


/**
 * Methods table.
 */

static PyMethodDef NvdecodeDataLoaderIterCoreBatchCM_methods[] = {
    {"__enter__", (PyCFunction)NvdecodeDataLoaderIterCoreBatchCM___enter__, METH_NOARGS,                "Enter a batch definition context."},
    {"__exit__",  (PyCFunction)NvdecodeDataLoaderIterCoreBatchCM___exit__,  METH_VARARGS|METH_KEYWORDS, "Exit the batch definition context and submit it."},
    {"setToken",  (PyCFunction)NvdecodeDataLoaderIterCoreBatchCM_setToken,  METH_VARARGS|METH_KEYWORDS, "Set or reset the batch token."},
    {"sample",    (PyCFunction)NvdecodeDataLoaderIterCoreBatchCM_sample,    METH_VARARGS|METH_KEYWORDS, "Create a new sample definition context within the batch definition context."},
    {NULL}  /* Sentinel */
};

static PyTypeObject NvdecodeDataLoaderIterCoreBatchCMType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "benzina.native.NvdecodeDataLoaderIterCoreBatchCM",       /* tp_name */
    sizeof(NvdecodeDataLoaderIterCoreBatchCM),                /* tp_basicsize */
    0,                                                        /* tp_itemsize */
    (destructor)NvdecodeDataLoaderIterCoreBatchCM_dealloc,    /* tp_dealloc */
    0,                                                        /* tp_print */
    0,                                                        /* tp_getattr */
    0,                                                        /* tp_setattr */
    0,                                                        /* tp_reserved */
    0,                                                        /* tp_repr */
    0,                                                        /* tp_as_number */
    0,                                                        /* tp_as_sequence */
    0,                                                        /* tp_as_mapping */
    0,                                                        /* tp_hash  */
    0,                                                        /* tp_call */
    0,                                                        /* tp_str */
    0,                                                        /* tp_getattro */
    0,                                                        /* tp_setattro */
    0,                                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,                  /* tp_flags */
    "NvdecodeDataLoaderIterCoreBatchCM object",               /* tp_doc */
    (traverseproc)NvdecodeDataLoaderIterCoreBatchCM_traverse, /* tp_traverse */
    (inquiry)NvdecodeDataLoaderIterCoreBatchCM_clear,         /* tp_clear */
    0,                                                        /* tp_richcompare */
    0,                                                        /* tp_weaklistoffset */
    0,                                                        /* tp_iter */
    0,                                                        /* tp_iternext */
    NvdecodeDataLoaderIterCoreBatchCM_methods,                /* tp_methods */
    0,                                                        /* tp_members */
    0,                                                        /* tp_getset */
    0,                                                        /* tp_base */
    0,                                                        /* tp_dict */
    0,                                                        /* tp_descr_get */
    0,                                                        /* tp_descr_set */
    0,                                                        /* tp_dictoffset */
    (initproc)NvdecodeDataLoaderIterCoreBatchCM_init,         /* tp_init */
    0,                                                        /* tp_alloc */
    (newfunc)NvdecodeDataLoaderIterCoreBatchCM_new,           /* tp_new */
    0,                                                        /* tp_free */
    0,                                                        /* tp_is_gc */
    0,                                                        /* tp_bases */
    0,                                                        /* tp_mro */
    0,                                                        /* tp_cache */
    0,                                                        /* tp_subclasses */
    0,                                                        /* tp_weaklist */
    0,                                                        /* tp_del */
    0,                                                        /* tp_version_tag */
    0,                                                        /* tp_finalize */
};
