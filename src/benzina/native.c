/* Includes */
#define  PY_SSIZE_T_CLEAN     /* So we get Py_ssize_t args. */
#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include "benzina/benzina.h"


/* Function Forward Declarations */
static PyObject* noop(PyObject* self, PyObject* args);


/**
 * Python Type Definitions.
 * 
 * Due to the voluminousness and uglyness of this code, it has been separated
 * into one Python class per .c/.h file.
 */

#include "./native_datasetcore.c"
#include "./native_nvdecodedataloaderitercore.c"
#include "./native_nvdecodedataloaderitercorebatchcm.c"
#include "./native_nvdecodedataloaderitercoresamplecm.c"



/* Function Definitions */

/**
 * @brief No-op.
 * 
 * Does nothing and returns None.
 */

static PyObject* noop(PyObject* self, PyObject* args){
	(void)self;
	(void)args;
	Py_INCREF(Py_None);
	return Py_None;
}


/**
 * Python Module Definition.
 */

static const char NATIVE_NOOP_DOC[] =
"No-op.";

static PyMethodDef native_module_methods[] = {
	{"noop", (PyCFunction)noop, 1, NATIVE_NOOP_DOC},
	{NULL},  /* Sentinel */
};

static const char NATIVE_MODULE_DOC[] =
"Benzina native library wrapper.";

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

PyMODINIT_FUNC PyInit_native(void){
	PyObject* m = NULL;
	
	if(benzinaInit() != 0){
		PyErr_SetString(PyExc_RuntimeError, "Could not initialize libbenzina!");
		return NULL;
	}
	
	m = PyModule_Create(&native_module_def);
	if(!m){return NULL;}
	
	#define ADDTYPE(T)                                       \
	    do{                                                  \
	        if(PyType_Ready(&T##Type) < 0){return NULL;}     \
	        Py_INCREF(&T##Type);                             \
	        PyModule_AddObject(m, #T, (PyObject*)&T##Type);  \
	    }while(0)
	ADDTYPE(DatasetCore);
	ADDTYPE(NvdecodeDataLoaderIterCore);
	ADDTYPE(NvdecodeDataLoaderIterCoreBatchCM);
	ADDTYPE(NvdecodeDataLoaderIterCoreSampleCM);
	#undef ADDTYPE
	
	return m;
}

