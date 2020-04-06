/* Includes */
#include "_C.h"
#include "structmember.h"
#include "benzina/benzina.h"




static void BenzinaFileViewObject_alloc            (PyTypeObject*      self,
                                                       Py_ssize_t         nitems){
    
}

static int  BenzinaFileViewObject_new              (PyTypeObject*      self,
                                                       PyObject*          args,
                                                       PyObject*          kwargs){
    return 0;
}

static int  BenzinaFileViewObject_init             (BenzinaViewObject* self,
                                                       PyObject*          args,
                                                       PyObject*          kwargs){
    return 0;
}

static void BenzinaFileViewObject_dealloc          (BenzinaViewObject* self){
    
}

static int  BenzinaFileViewObject_getbufferproc    (BenzinaViewObject* exporter,
                                                       Py_buffer*         view,
                                                       int                flags){
    return PyBuffer_FillInfo(view, (PyObject*)exporter, (void*)exporter->data,
                             exporter->length, 1, flags);
}

static void BenzinaFileViewObject_releasebufferproc(BenzinaViewObject* exporter,
                                                       Py_buffer*         view){
    
}

static int  BenzinaFileViewObject_traverse         (BenzinaViewObject* self,
                                                       visitproc          visit,
                                                       void*              arg){
    Py_VISIT(self->child);
    Py_VISIT(self->next);
    return 0;
}

static int  BenzinaFileViewObject_clear            (BenzinaViewObject* self){
    Py_CLEAR(self->child);
    Py_CLEAR(self->next);
    return 0;
}

/*
PyMemberDef

    Structure which describes an attribute of a type which corresponds to a C struct member. Its fields are:
    Field 	C Type 	Meaning
    name 	char * 	name of the member
    type 	int 	the type of the member in the C struct
    offset 	Py_ssize_t 	the offset in bytes that the member is located on the type’s object struct
    flags 	int 	flag bits indicating if the field should be read-only or writable
    doc 	char * 	points to the contents of the docstring

    type can be one of many T_ macros corresponding to various C types. When the member is accessed in Python, it will be converted to the equivalent Python type.
    Macro name 	C type
    T_SHORT 	short
    T_INT 	int
    T_LONG 	long
    T_FLOAT 	float
    T_DOUBLE 	double
    T_STRING 	char *
    T_OBJECT 	PyObject *
    T_OBJECT_EX 	PyObject *
    T_CHAR 	char
    T_BYTE 	char
    T_UBYTE 	unsigned char
    T_UINT 	unsigned int
    T_USHORT 	unsigned short
    T_ULONG 	unsigned long
    T_BOOL 	char
    T_LONGLONG 	long long
    T_ULONGLONG 	unsigned long long
    T_PYSSIZET 	Py_ssize_t

    T_OBJECT and T_OBJECT_EX differ in that T_OBJECT returns None if the member is NULL and T_OBJECT_EX raises an AttributeError. Try to use T_OBJECT_EX over T_OBJECT because T_OBJECT_EX handles use of the del statement on that attribute more correctly than T_OBJECT.

    flags can be 0 for write and read access or READONLY for read-only access. Using T_STRING for type implies READONLY. Only T_OBJECT and T_OBJECT_EX members can be deleted. (They are set to NULL).
*/

static PyMemberDef BenzinaFileViewObject_members[] = {
    {NULL},
};

static PyMethodDef BenzinaFileViewObject_methods[] = {
    {NULL},
};

/**
 * typedef struct PyGetSetDef {
 *     char *name;
 *     getter get;
 *     setter set;
 *     char *doc;
 *     void *closure;
 * } PyGetSetDef;
 */

static PyGetSetDef BenzinaFileViewObject_getset[] = {
    {NULL},
};

static PyBufferProcs BenzinaFileViewObject_bufferprocs = {
    (getbufferproc)    BenzinaFileViewObject_getbufferproc,
    (releasebufferproc)BenzinaFileViewObject_releasebufferproc,
};

BENZINA_ATTRIBUTE_HIDDEN PyTypeObject BenzinaFileViewType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name         = "benzina._native._C.FileView",
    .tp_doc          = "FileView object documentation",
    .tp_basicsize    = sizeof(BenzinaFileViewObject),
    .tp_flags        = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_BASETYPE,
    
    .tp_alloc        =    (allocfunc)BenzinaFileViewObject_alloc,
    .tp_new          =      (newfunc)BenzinaFileViewObject_new,
    .tp_init         =     (initproc)BenzinaFileViewObject_init,
    .tp_dealloc      =   (destructor)BenzinaFileViewObject_dealloc,
    .tp_as_buffer    =              &BenzinaFileViewObject_bufferprocs,
    .tp_methods      =               BenzinaFileViewObject_methods,
    .tp_members      =               BenzinaFileViewObject_members,
    .tp_getset       =               BenzinaFileViewObject_getset,
    .tp_traverse     = (traverseproc)BenzinaFileViewObject_traverse,
    .tp_clear        =      (inquiry)BenzinaFileViewObject_clear,
};

