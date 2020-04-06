/* Includes */
#include "_C.h"
#include "benzina/benzina.h"



/* Function Definitions */



/**
 * Python Module Definition.
 */

static PyModuleDef _C_module_def = {
    PyModuleDef_HEAD_INIT,
    "_C",
    "benzina._native._C bindings",
    -1,                    /* m_size */
    NULL,                  /* m_methods */
    NULL,                  /* m_reload */
    NULL,                  /* m_traverse */
    NULL,                  /* m_clear */
    NULL,                  /* m_free */
};

/**
 * Initialization.
 */

PyMODINIT_FUNC PyInit__C(void){
    PyObject* m;
    
    if(benz_init() != 0){
        PyErr_SetString(PyExc_RuntimeError, "Could not initialize libbenzina!");
        return NULL;
    }
    
    m = PyModule_Create(&_C_module_def);
    if(!m){return NULL;}
    
    #define ADDTYPE(T)                                                \
        do{                                                           \
            if(PyType_Ready(&Benzina##T##Type) < 0){return NULL;}     \
            Py_INCREF(&Benzina##T##Type);                             \
            PyModule_AddObject(m, #T, (PyObject*)&Benzina##T##Type);  \
        }while(0)
    ADDTYPE(View);
    ADDTYPE(BoxView);
    ADDTYPE(FullBoxView);
    ADDTYPE(FileView);
    #undef ADDTYPE
    
    return m;
}

