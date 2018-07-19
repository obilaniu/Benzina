/* Includes */
#define  PY_SSIZE_T_CLEAN  /* So we get Py_ssize_t args. */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include <structmember.h>
#include <stdint.h>
#include "native.h"


/* Defines */



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
	PyObject* root=NULL;
	
	static char *kwargsList[] = {"root", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "U", kwargsList, &root)){
		return -1;
	}
	
	root = PyUnicode_EncodeFSDefault(root);
	if(!root){
		return -1;
	}
	if(!PyBytes_Check(root)){
		Py_DECREF(root);
		return -1;
	}
	
	if(benzinaDatasetNew(&self->dataset, PyBytes_AsString(root)) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error during creation of underlying BENZINA_DATASET* "
		                "object. Check that the path is correct and has all of "
		                "the required files.");
		Py_DECREF(root);
		return -1;
	}
	
	Py_DECREF(root);
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
 * @brief Getter for shape.
 * @return The shape, as a tuple (h,w).
 */

static PyObject* DatasetCore_getshape (DatasetCore* self,
                                              void*               closure){
	size_t    h,w;
	PyObject* hObj, *wObj, *tObj;
	
	if(benzinaDatasetGetShape(self->dataset, &w, &h) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not obtain shape of dataset for unknown reasons.");
		return NULL;
	}
	
	hObj = PyLong_FromSize_t(h);
	wObj = PyLong_FromSize_t(w);
	if(!hObj || !wObj){
		Py_XDECREF(hObj);
		Py_XDECREF(wObj);
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not create a Python integer!");
		return NULL;
	}
	
	tObj = PyTuple_Pack(2, hObj, wObj);
	if(!tObj){
		Py_DECREF(hObj);
		Py_DECREF(wObj);
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not create a Python tuple!");
		return NULL;
	}
	
	return tObj;
}

/**
 * Table of getter-setters.
 * 
 * We only have getters.
 */

static PyGetSetDef       DatasetCore_getsetters[] = {
    {"length", (getter)DatasetCore_getlength, 0, "Length of Dataset",             NULL},
    {"shape",  (getter)DatasetCore_getshape,  0, "Coded shape of dataset images", NULL},
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

/**
 * @brief Implementation of __getitem__.
 * 
 * @return A tuple (i, off, len) indicating:
 *           - The index this image was fetched at.
 *           - The offset into data.bin
 *           - The length of the slice into it.
 */

static PyObject*         DatasetCore___getitem__(DatasetCore* self,
                                                        Py_ssize_t          i){
	size_t    off, len;
	PyObject* lenObj, *iObj, *offObj, *tObj;
	
	if(benzinaDatasetGetElement(self->dataset, i, &off, &len) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not read element of dataset for unknown reasons.");
		return NULL;
	}
	
	iObj   = PyLong_FromSsize_t(i);
	offObj = PyLong_FromSize_t(off);
	lenObj = PyLong_FromSize_t(len);
	if(!iObj || !offObj || !lenObj){
		Py_XDECREF(iObj);
		Py_XDECREF(offObj);
		Py_XDECREF(lenObj);
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not create a Python integer!");
		return NULL;
	}
	
	tObj = PyTuple_Pack(3, iObj, offObj, lenObj);
	if(!tObj){
		Py_DECREF(iObj);
		Py_DECREF(offObj);
		Py_DECREF(lenObj);
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not create a Python tuple!");
		return NULL;
	}
	
	return tObj;
}

static PySequenceMethods DatasetCore_as_seq_methods = {
    (lenfunc)DatasetCore___len__,          /* sq_length */
    0,                                            /* sq_concat */
    0,                                            /* sq_repeat */
    (ssizeargfunc)DatasetCore___getitem__, /* sq_item */
};

static PyTypeObject DatasetCoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "native.DatasetCore",            /* tp_name */
    sizeof(DatasetCore),             /* tp_basicsize */
    0,                                      /* tp_itemsize */
    (destructor)DatasetCore_dealloc, /* tp_dealloc */
    0,                                      /* tp_print */
    0,                                      /* tp_getattr */
    0,                                      /* tp_setattr */
    0,                                      /* tp_reserved */
    0,                                      /* tp_repr */
    0,                                      /* tp_as_number */
    &DatasetCore_as_seq_methods,     /* tp_as_sequence */
    0,                                      /* tp_as_mapping */
    0,                                      /* tp_hash  */
    0,                                      /* tp_call */
    0,                                      /* tp_str */
    0,                                      /* tp_getattro */
    0,                                      /* tp_setattro */
    0,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
    "DatasetCore object",            /* tp_doc */
    0,                                      /* tp_traverse */
    0,                                      /* tp_clear */
    0,                                      /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    0,                                      /* tp_iter */
    0,                                      /* tp_iternext */
    0,                                      /* tp_methods */
    0,                                      /* tp_members */
    DatasetCore_getsetters,          /* tp_getset */
    0,                                      /* tp_base */
    0,                                      /* tp_dict */
    0,                                      /* tp_descr_get */
    0,                                      /* tp_descr_set */
    0,                                      /* tp_dictoffset */
    (initproc)DatasetCore_init,      /* tp_init */
    0,                                      /* tp_alloc */
    DatasetCore_new,                 /* tp_new */
};
