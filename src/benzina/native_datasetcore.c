/* Includes */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include <stdint.h>
#include "./native_datasetcore.h"



/* Python API Function Definitions */

/**
 * @brief Slot tp_dealloc
 */

static void      DatasetCore_dealloc  (DatasetCore* self){
	benzinaDatasetFree(self->dataset);
	self->dataset = NULL;
	Py_TYPE(self)->tp_free(self);
}

/**
 * @brief Slot tp_new
 */

static PyObject* DatasetCore_new      (PyTypeObject* type,
                                       PyObject*     args,
                                       PyObject*     kwargs){
	(void)args;
	(void)kwargs;
	
	DatasetCore* self = (DatasetCore*)type->tp_alloc(type, 0);
	
	if(self){
		self->dataset = NULL;
	}
	
	return (PyObject*)self;
}

/**
 * @brief Slot tp_init
 */

static int       DatasetCore_init     (DatasetCore* self,
                                       PyObject*           args,
                                       PyObject*           kwargs){
	PyObject* file  = NULL;
	uint64_t length = 0;
	
	static char *kwargsList[] = {"file", "length", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "Uk", kwargsList, &file, &length)){
		return -1;
	}
	
	file = PyUnicode_EncodeFSDefault(file);
	if(!file){
		return -1;
	}
	if(!PyBytes_Check(file)){
		Py_DECREF(file);
		return -1;
	}
	
	if(benzinaDatasetNew(&self->dataset, PyBytes_AsString(file), &length) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error during creation of underlying BENZINA_DATASET* "
		                "object. Check that the path is correct and has all of "
		                "the required files.");
		Py_DECREF(file);
		return -1;
	}
	
	Py_DECREF(file);
	return 0;
}

/**
 * @brief Getter for length.
 */

static PyObject* DatasetCore_getlength(DatasetCore* self,
                                              void*               closure){
	size_t length;
	
	if(benzinaDatasetGetLength(self->dataset, &length) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not obtain length of dataset for unknown reasons.");
		return NULL;
	}
	
	return PyLong_FromSize_t(length);
}

/**
 * Table of getter-setters.
 * 
 * We only have getters.
 */

static PyGetSetDef       DatasetCore_getsetters[] = {
    {"length", (getter)DatasetCore_getlength, 0, "Length of Dataset",             NULL},
    {NULL}  /* Sentinel */
};

/**
 * @brief Implementation of __len__
 */

static Py_ssize_t        DatasetCore___len__(DatasetCore* self){
	size_t length;
	
	if(benzinaDatasetGetLength(self->dataset, &length) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not obtain length of dataset for unknown reasons.");
		return -1;
	}
	
	return length;
}

static PySequenceMethods DatasetCore_as_seq_methods = {
    (lenfunc)DatasetCore___len__,    /* sq_length */
    0,                               /* sq_concat */
    0,                               /* sq_repeat */
};

static PyTypeObject DatasetCoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "benzina.native.DatasetCore",    /* tp_name */
    sizeof(DatasetCore),             /* tp_basicsize */
    0,                               /* tp_itemsize */
    (destructor)DatasetCore_dealloc, /* tp_dealloc */
    0,                               /* tp_print */
    0,                               /* tp_getattr */
    0,                               /* tp_setattr */
    0,                               /* tp_reserved */
    0,                               /* tp_repr */
    0,                               /* tp_as_number */
    &DatasetCore_as_seq_methods,     /* tp_as_sequence */
    0,                               /* tp_as_mapping */
    0,                               /* tp_hash  */
    0,                               /* tp_call */
    0,                               /* tp_str */
    0,                               /* tp_getattro */
    0,                               /* tp_setattro */
    0,                               /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,              /* tp_flags */
    "DatasetCore object",            /* tp_doc */
    0,                               /* tp_traverse */
    0,                               /* tp_clear */
    0,                               /* tp_richcompare */
    0,                               /* tp_weaklistoffset */
    0,                               /* tp_iter */
    0,                               /* tp_iternext */
    0,                               /* tp_methods */
    0,                               /* tp_members */
    DatasetCore_getsetters,          /* tp_getset */
    0,                               /* tp_base */
    0,                               /* tp_dict */
    0,                               /* tp_descr_get */
    0,                               /* tp_descr_set */
    0,                               /* tp_dictoffset */
    (initproc)DatasetCore_init,      /* tp_init */
    0,                               /* tp_alloc */
    DatasetCore_new,                 /* tp_new */
};
