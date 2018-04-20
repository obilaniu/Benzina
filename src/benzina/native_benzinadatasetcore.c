/* Includes */
#define  PY_SSIZE_T_CLEAN  /* So we get Py_ssize_t args. */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include "structmember.h"
#include <stdint.h>
#include "benzina.h"


typedef struct {
    PyObject_HEAD
    BENZINA_DATASET* dset;
} BenzinaDatasetCore;

static void
BenzinaDatasetCore_dealloc(BenzinaDatasetCore* self)
{
    Py_XDECREF(self->first);
    Py_XDECREF(self->last);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
BenzinaDatasetCore_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BenzinaDatasetCore *self;

    self = (BenzinaDatasetCore *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->first = PyUnicode_FromString("");
        if (self->first == NULL) {
            Py_DECREF(self);
            return NULL;
        }

        self->last = PyUnicode_FromString("");
        if (self->last == NULL) {
            Py_DECREF(self);
            return NULL;
        }

        self->number = 0;
    }

    return (PyObject *)self;
}

static int
BenzinaDatasetCore_init(BenzinaDatasetCore *self, PyObject *args, PyObject *kwds)
{
    PyObject *first=NULL, *last=NULL, *tmp;

    static char *kwlist[] = {"first", "last", "number", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|SSi", kwlist,
                                      &first, &last,
                                      &self->number))
        return -1;

    if (first) {
        tmp = self->first;
        Py_INCREF(first);
        self->first = first;
        Py_DECREF(tmp);
    }

    if (last) {
        tmp = self->last;
        Py_INCREF(last);
        self->last = last;
        Py_DECREF(tmp);
    }

    return 0;
}

static PyMemberDef BenzinaDatasetCore_members[] = {
    {"number", T_INT, offsetof(BenzinaDatasetCore, number), 0,
     "BenzinaDatasetCore number"},
    {NULL}  /* Sentinel */
};

static PyObject *
BenzinaDatasetCore_getfirst(BenzinaDatasetCore *self, void *closure)
{
    Py_INCREF(self->first);
    return self->first;
}

static int
BenzinaDatasetCore_setfirst(BenzinaDatasetCore *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the first attribute");
        return -1;
    }

    if (! PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "The first attribute value must be a string");
        return -1;
    }

    Py_DECREF(self->first);
    Py_INCREF(value);
    self->first = value;

    return 0;
}

static PyObject *
BenzinaDatasetCore_getlast(BenzinaDatasetCore *self, void *closure)
{
    Py_INCREF(self->last);
    return self->last;
}

static int
BenzinaDatasetCore_setlast(BenzinaDatasetCore *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the last attribute");
        return -1;
    }

    if (! PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "The last attribute value must be a string");
        return -1;
    }

    Py_DECREF(self->last);
    Py_INCREF(value);
    self->last = value;

    return 0;
}

static PyGetSetDef BenzinaDatasetCore_getseters[] = {
    {"first",
     (getter)BenzinaDatasetCore_getfirst, (setter)BenzinaDatasetCore_setfirst,
     "first name",
     NULL},
    {"last",
     (getter)BenzinaDatasetCore_getlast, (setter)BenzinaDatasetCore_setlast,
     "last name",
     NULL},
    {NULL}  /* Sentinel */
};

static PyObject *
BenzinaDatasetCore_name(BenzinaDatasetCore* self)
{
    return PyUnicode_FromFormat("%S %S", self->first, self->last);
}

static PyMethodDef BenzinaDatasetCore_methods[] = {
    {"name", (PyCFunction)BenzinaDatasetCore_name, METH_NOARGS,
     "Return the name, combining the first and last name"
    },
    {NULL}  /* Sentinel */
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
    0,                                      /* tp_as_sequence */
    0,                                      /* tp_as_mapping */
    0,                                      /* tp_hash  */
    0,                                      /* tp_call */
    0,                                      /* tp_str */
    0,                                      /* tp_getattro */
    0,                                      /* tp_setattro */
    0,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,                /* tp_flags */
    "BenzinaDatasetCore objects",           /* tp_doc */
    0,                                      /* tp_traverse */
    0,                                      /* tp_clear */
    0,                                      /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    0,                                      /* tp_iter */
    0,                                      /* tp_iternext */
    BenzinaDatasetCore_methods,             /* tp_methods */
    BenzinaDatasetCore_members,             /* tp_members */
    BenzinaDatasetCore_getseters,           /* tp_getset */
    0,                                      /* tp_base */
    0,                                      /* tp_dict */
    0,                                      /* tp_descr_get */
    0,                                      /* tp_descr_set */
    0,                                      /* tp_dictoffset */
    (initproc)BenzinaDatasetCore_init,      /* tp_init */
    0,                                      /* tp_alloc */
    BenzinaDatasetCore_new,                 /* tp_new */
};


#if 0
PyObject*          ret;
unsigned long      crc32c;
unsigned long      arg_crc32c = -1;
const char*        arg_bufptr = NULL;
Py_ssize_t         arg_buflen = -1;
unsigned long long arg_len    = -1;
unsigned long long arg_off    =  0;


if(!PyArg_ParseTuple(args, "|kz#KK",
                     &arg_crc32c,
                     &arg_bufptr, &arg_buflen,
                     &arg_len,
                     &arg_off)){
	PyErr_SetString(PyExc_RuntimeError, "Failed to parse arguments!");
	return NULL;
}
if      (arg_len != -1 && arg_off+arg_len > arg_buflen){
	PyErr_SetString(PyExc_ValueError,   "Buffer overflow (offset + length beyond end of buffer)!");
	return NULL;
}else if(arg_len == -1 && arg_off > arg_buflen){
	PyErr_SetString(PyExc_ValueError,   "Buffer overflow (offset greater than buffer length)!");
	return NULL;
}else if(arg_len == -1){
	arg_len = arg_buflen-arg_off;
}

crc32c = doCRC32C(arg_crc32c,
                  arg_bufptr+arg_off,
                  arg_len);
ret = PyLong_FromLong(crc32c);

if(!ret){
	PyErr_SetString(PyExc_RuntimeError, "Failed to allocate Python int!");
}
return ret;
#endif
