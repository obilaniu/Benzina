/* Include Guard */
#ifndef SRC_BENZINA__NATIVE__C__C_H
#define SRC_BENZINA__NATIVE__C__C_H


/**
 * Includes
 */

#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include <stdint.h>
#include <string.h>
#include "benzina/endian.h"
#include "benzina/inline.h"
#include "benzina/visibility.h"
#include "benzina/ptrops.h"


/* Data Structures Forward Declarations and Typedefs */
typedef struct BenzinaViewObject         BenzinaViewObject;
typedef struct BenzinaFileViewObject     BenzinaFileViewObject;
typedef struct BenzinaBoxViewObject      BenzinaBoxViewObject;
typedef struct BenzinaFullBoxViewObject  BenzinaFullBoxViewObject;



/* Data Structure Definitions */

/**
 * @brief BenzinaView Python Object
 * 
 * Contains the critical "bypass" attributes as isolated C pointers for quick
 * GIL-less navigation in C code.
 * 
 * The layout of the struct is optimized such that:
 * 
 *   - The Python code touches the first cacheline in read/write mode almost
 *     exclusively (in particular, the reference count at offset 0x00), while
 *   - The C code touches the second cacheline in read mode almost exclusively
 *     (in particular, the data, length, next, child, type and flagver attributes)
 * 
 * The hierarchy of inheritance is as follows:
 * 
 *   - View (Abstract Base Class)
 *     - FileView
 *     - BoxView
 *       - FullBoxView
 * 
 * Although the View class preallocates storage for most of the important fields
 * of its subclasses, it does not expose those fields at the Python level itself.
 * Instead it relies on the subclasses to make those fields visible.
 */

struct BenzinaViewObject{
    PyObject_HEAD                        /* 0x00 Python object base */
    PyObject*                dict;       /* 0x10 Pointer to __dict__    Python object */
    PyObject*                weakref;    /* 0x18 Pointer to __weakref__ Python object */
    PyObject*                exporter;   /* 0x20 Pointer to exporter    Python object */
    BenzinaViewObject*       parent;     /* 0x28 Pointer to parent      Python object */
    uint8_t                  uuid[16];   /* 0x30 UUID box extended type. Copied
                                                 directly from box. */
    
    const void*              data;       /* 0x40 View data */
    Py_ssize_t               length;     /* 0x48 View length */
    BenzinaViewObject*       next;       /* 0x50 Next box at same level */
    BenzinaViewObject*       child;      /* 0x58 First child box */
    uint32_t                 type;       /* 0x60 Box type. Copied directly from box. */
    uint32_t                 flagver;    /* 0x64 Box flags. Copied directly from box. */
};

struct BenzinaBoxViewObject{
    BenzinaViewObject view;
    const void*       payload;
};

struct BenzinaFullBoxViewObject{
    BenzinaBoxViewObject view;
    const void*          dummy;
};

struct BenzinaFileViewObject{
    BenzinaViewObject view;
    Py_buffer         buffer;
};


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Declare types */
#ifndef DECLAREPYTYPE
#define DECLAREPYTYPE(T)                \
    BENZINA_ATTRIBUTE_HIDDEN extern PyTypeObject Benzina##T##Type
#endif
DECLAREPYTYPE(View);
DECLAREPYTYPE(BoxView);
DECLAREPYTYPE(FullBoxView);
DECLAREPYTYPE(FileView);
#undef  DECLAREPYTYPE


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

