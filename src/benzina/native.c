/* Includes */
#define  PY_SSIZE_T_CLEAN  /* So we get Py_ssize_t args. */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include "benzina.h"


/* Python Method Forward Declarations */
PyObject* noop(PyObject* self, PyObject* args);



/* Python Method Definitions */

/**
 * @brief No-op.
 * 
 * Does nothing and returns None.
 */

PyObject* noop(PyObject* self, PyObject* args){
	(void)self;
	(void)args;
	Py_INCREF(Py_None);
	return Py_None;
}



/**
 * Python Type Definitions.
 * 
 * Due to the voluminousness and uglyness of this code, it has been separated
 * into one Python type per C file.
 */

#include "./native_benzinadatasetcore.c"
#include "./native_benzinaloaderitercore.c"



/* String Constants. */
static const char NATIVE_MODULE_DOC[] =
"Benzina native library wrapper.";
static const char NATIVE_NOOP_DOC[] =
"No-op.";



/* Python Module Definition */
static PyMethodDef native_module_methods[] = {
    {"noop", (PyCFunction)noop, 1, NATIVE_NOOP_DOC},
};
static PyModuleDef native_module_def = {
	PyModuleDef_HEAD_INIT,
	"native",              /* m_name */
	NATIVE_MODULE_DOC,     /* m_doc */
	-1,                    /* m_size */
	native_module_methods, /* m_methods */
	NULL,                  /* m_reload */
	NULL,                  /* m_traverse */
	NULL,                  /* m_clear */
	NULL,                  /* m_free */
};

/**
 * @brief Initialize Python module.
 */

PyMODINIT_FUNC PyInit_native(void){
	PyObject* m = NULL;
	
	/* Initialize libbenzina. */
	if(benzinaInit() != 0){return NULL;}
	
	/* Finish readying Python wrapper types. */
	if(PyType_Ready(&BenzinaDatasetCoreType   ) < 0){return NULL;};
	if(PyType_Ready(&BenzinaLoaderIterCoreType) < 0){return NULL;};
	
	/* Create the module. */
	m = PyModule_Create(&native_module_def);
	if(!m){return NULL;}
	
	/* Register Python wrapper types. */
	Py_INCREF(&BenzinaDatasetCoreType);
	Py_INCREF(&BenzinaLoaderIterCoreType);
	PyModule_AddObject(m, "BenzinaDatasetCore",    (PyObject*)&BenzinaDatasetCoreType);
	PyModule_AddObject(m, "BenzinaLoaderIterCore", (PyObject*)&BenzinaLoaderIterCoreType);
	
	/* Return module. */
	return m;
}

