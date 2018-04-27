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

static void      BenzinaDatasetCore_dealloc  (BenzinaDatasetCore* self){
	benzinaDatasetFree(self->dset);
	Py_TYPE(self)->tp_free(self);
}

/**
 * @brief Slot tp_new
 */

static PyObject* BenzinaDatasetCore_new      (PyTypeObject* type,
                                              PyObject*     args,
                                              PyObject*     kwargs){
	BenzinaDatasetCore* self = (BenzinaDatasetCore*)type->tp_alloc(type, 0);
	
	self->dset = NULL;
	
	return (PyObject*)self;
}

/**
 * @brief Slot tp_init
 */

static int       BenzinaDatasetCore_init     (BenzinaDatasetCore* self,
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
	
	if(benzinaDatasetNew(&self->dset, PyBytes_AsString(root)) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error during creation of underlying BENZINA_DATASET* "
		                "object. Check that the path is correct and has all of "
		                "the required files.");
		Py_DECREF(root);
		return -1;
	}
	
	return 0;
}

/**
 * @brief Getter for length.
 */

static PyObject* BenzinaDatasetCore_getlength(BenzinaDatasetCore* self,
                                              void*               closure){
	size_t length;
	
	if(benzinaDatasetGetLength(self->dset, &length) != 0){
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

static PyObject* BenzinaDatasetCore_getshape (BenzinaDatasetCore* self,
                                              void*               closure){
	size_t    h,w;
	PyObject* hObj, *wObj, *tObj;
	
	if(benzinaDatasetGetShape(self->dset, &w, &h) != 0){
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

static PyGetSetDef       BenzinaDatasetCore_getsetters[] = {
    {"length", (getter)BenzinaDatasetCore_getlength, 0, "Length of Dataset",       NULL},
    {"shape",  (getter)BenzinaDatasetCore_getshape,  0, "Shape of dataset images", NULL},
    {NULL}  /* Sentinel */
};

/**
 * @brief Implementation of __len__
 */

static Py_ssize_t        BenzinaDatasetCore___len__(BenzinaDatasetCore* self){
	size_t length;
	
	if(benzinaDatasetGetLength(self->dset, &length) != 0){
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

static PyObject*         BenzinaDatasetCore___getitem__(BenzinaDatasetCore* self,
                                                        Py_ssize_t          i){
	size_t    off, len;
	PyObject* lenObj, *iObj, *offObj, *tObj;
	
	if(benzinaDatasetGetElement(self->dset, i, &off, &len) != 0){
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

static PySequenceMethods BenzinaDatasetCore_as_seq_methods = {
    (lenfunc)BenzinaDatasetCore___len__,          /* sq_length */
    0,                                            /* sq_concat */
    0,                                            /* sq_repeat */
    (ssizeargfunc)BenzinaDatasetCore___getitem__, /* sq_item */
};

static PyTypeObject BenzinaDatasetCoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "native.BenzinaDatasetCore",            /* tp_name */
    sizeof(BenzinaDatasetCore),             /* tp_basicsize */
    0,                                      /* tp_itemsize */
    (destructor)BenzinaDatasetCore_dealloc, /* tp_dealloc */
    0,                                      /* tp_print */
    0,                                      /* tp_getattr */
    0,                                      /* tp_setattr */
    0,                                      /* tp_reserved */
    0,                                      /* tp_repr */
    0,                                      /* tp_as_number */
    &BenzinaDatasetCore_as_seq_methods,     /* tp_as_sequence */
    0,                                      /* tp_as_mapping */
    0,                                      /* tp_hash  */
    0,                                      /* tp_call */
    0,                                      /* tp_str */
    0,                                      /* tp_getattro */
    0,                                      /* tp_setattro */
    0,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /* tp_flags */
    "BenzinaDatasetCore object",            /* tp_doc */
    0,                                      /* tp_traverse */
    0,                                      /* tp_clear */
    0,                                      /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    0,                                      /* tp_iter */
    0,                                      /* tp_iternext */
    0,                                      /* tp_methods */
    0,                                      /* tp_members */
    BenzinaDatasetCore_getsetters,          /* tp_getset */
    0,                                      /* tp_base */
    0,                                      /* tp_dict */
    0,                                      /* tp_descr_get */
    0,                                      /* tp_descr_set */
    0,                                      /* tp_dictoffset */
    (initproc)BenzinaDatasetCore_init,      /* tp_init */
    0,                                      /* tp_alloc */
    BenzinaDatasetCore_new,                 /* tp_new */
};
