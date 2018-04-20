/* Includes */
#define  PY_SSIZE_T_CLEAN  /* So we get Py_ssize_t args. */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include "structmember.h"
#include <stdint.h>
#include "benzina.h"


typedef struct {
    PyObject_HEAD
    BENZINA_LOADER_ITER* iter;
} BenzinaLoaderIterCore;

static void
BenzinaLoaderIterCore_dealloc(BenzinaLoaderIterCore* self)
{
    Py_XDECREF(self->first);
    Py_XDECREF(self->last);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
BenzinaLoaderIterCore_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BenzinaLoaderIterCore *self;

    self = (BenzinaLoaderIterCore *)type->tp_alloc(type, 0);
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
BenzinaLoaderIterCore_init(BenzinaLoaderIterCore *self, PyObject *args, PyObject *kwds)
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

static PyMemberDef BenzinaLoaderIterCore_members[] = {
    {"number", T_INT, offsetof(BenzinaLoaderIterCore, number), 0,
     "BenzinaLoaderIterCore number"},
    {NULL}  /* Sentinel */
};

static PyObject *
BenzinaLoaderIterCore_getfirst(BenzinaLoaderIterCore *self, void *closure)
{
    Py_INCREF(self->first);
    return self->first;
}

static int
BenzinaLoaderIterCore_setfirst(BenzinaLoaderIterCore *self, PyObject *value, void *closure)
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
BenzinaLoaderIterCore_getlast(BenzinaLoaderIterCore *self, void *closure)
{
    Py_INCREF(self->last);
    return self->last;
}

static int
BenzinaLoaderIterCore_setlast(BenzinaLoaderIterCore *self, PyObject *value, void *closure)
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

static PyGetSetDef BenzinaLoaderIterCore_getseters[] = {
    {"first",
     (getter)BenzinaLoaderIterCore_getfirst, (setter)BenzinaLoaderIterCore_setfirst,
     "first name",
     NULL},
    {"last",
     (getter)BenzinaLoaderIterCore_getlast, (setter)BenzinaLoaderIterCore_setlast,
     "last name",
     NULL},
    {NULL}  /* Sentinel */
};

static PyObject *
BenzinaLoaderIterCore_name(BenzinaLoaderIterCore* self)
{
    return PyUnicode_FromFormat("%S %S", self->first, self->last);
}

static PyMethodDef BenzinaLoaderIterCore_methods[] = {
    {"name", (PyCFunction)BenzinaLoaderIterCore_name, METH_NOARGS,
     "Return the name, combining the first and last name"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject BenzinaLoaderIterCoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "native.BenzinaLoaderIterCore",             /* tp_name */
    sizeof(BenzinaLoaderIterCore),              /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)BenzinaLoaderIterCore_dealloc,  /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_reserved */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash  */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,                    /* tp_flags */
    "BenzinaLoaderIterCore objects",            /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    BenzinaLoaderIterCore_methods,              /* tp_methods */
    BenzinaLoaderIterCore_members,              /* tp_members */
    BenzinaLoaderIterCore_getseters,            /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)BenzinaLoaderIterCore_init,       /* tp_init */
    0,                                          /* tp_alloc */
    BenzinaLoaderIterCore_new,                  /* tp_new */
};
