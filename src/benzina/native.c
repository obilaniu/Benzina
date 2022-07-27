/* Includes */
#include <Python.h>           /* Because of "reasons", the Python header must be first. */
#include "benzina/benzina-old.h"



/**
 * xorshift128+: A fast, high-quality PRNG.
 * 
 * Implementation taken from
 *     https://en.wikipedia.org/wiki/Xorshift#xorshift+
 */
typedef struct XORSHIFT128PLUS{uint64_t S[2];} XORSHIFT128PLUS;

static void     xorshift128plus_seed(XORSHIFT128PLUS *S, uint64_t seed){
    S->S[0] =  seed;
    S->S[1] = ~seed;
}
static uint64_t xorshift128plus(XORSHIFT128PLUS *S){
    uint64_t p = S->S[0];
    uint64_t q = S->S[1];
    S->S[0] = q;
    p ^= p << 23;
    p ^= p >> 17;
    p ^= q ^ (q >> 26);
    S->S[1] = p;
    return p + q;
}
static double xorshift128plus_u01(XORSHIFT128PLUS *S){
    return xorshift128plus(S) * 5.421010862427522e-20;
}
static double xorshift128plus_uab(XORSHIFT128PLUS *S, double a, double b){
    double f = xorshift128plus_u01(S);
    return (1-f)*a + f*b;
}



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
 * @brief Similarity Transform.
 * 
 * Randomly draw a similarity transform from within the given bounds.
 */

static PyObject* similarity(PyObject* self, PyObject* args){
    (void)self;
    
    XORSHIFT128PLUS S;
    unsigned long long seed=0;
    int    autoscale=0;
    double ow=0, oh=0,     iw=0, ih=0;
    double scalelo=1,      scalehi   =1,   scale;
    double rotationlo=0,   rotationhi=0,   rotation;
    double txlo=0, txhi=0, tylo=0, tyhi=0, tx, ty;
    double flipx=1,        flipy=1;
    
    if(!PyArg_ParseTuple(args, "ddddddddddddddKp",
                         &ow, &oh,     &iw, &ih,
                         &scalelo,     &scalehi,
                         &rotationlo,  &rotationhi,
                         &txlo, &txhi, &tylo, &tyhi,
                         &flipx,       &flipy,
                         &seed,        &autoscale)){
        PyErr_SetString(PyExc_RuntimeError,
                        "Invalid arguments!");
        return NULL;
    }
    
    xorshift128plus_seed(&S, seed);
    scale    = exp(xorshift128plus_uab(&S, log(scalelo), log(scalehi)));
    rotation = xorshift128plus_uab(&S, rotationlo, rotationhi);
    tx       = xorshift128plus_uab(&S, txlo,       txhi);
    ty       = xorshift128plus_uab(&S, tylo,       tyhi);
    flipx    = xorshift128plus_u01(&S) < flipx ? -1 : +1;
    flipy    = xorshift128plus_u01(&S) < flipy ? -1 : +1;
    
    double T_o_y = (oh-1)/2;
    double T_o_x = (ow-1)/2;
    double S_y = flipy/scale;
    double S_x = flipx/scale;
    if(autoscale){
        S_y *= ih/oh;
        S_x *= iw/ow;
    }
    double sR    = sin(rotation);
    double cR    = cos(rotation);
    double T_i_y = (ih-1)/2;
    double T_i_x = (iw-1)/2;
    double H[3][3] = {
        {S_x*+cR, S_y*+sR, (S_x*+cR)*-T_o_x + (S_y*+sR)*-T_o_y + tx+T_i_x},
        {S_x*-sR, S_y*+cR, (S_x*-sR)*-T_o_x + (S_y*+cR)*-T_o_y + ty+T_i_y},
        {      0,       0,                                              1},
    };
    return Py_BuildValue("ddddddddd", H[0][0], H[0][1], H[0][2],
                                      H[1][0], H[1][1], H[1][2],
                                      H[2][0], H[2][1], H[2][2]);
}


/**
 * Python Module Definition.
 */

static const char NATIVE_NOOP_DOC[] =
"No-op.";
static const char NATIVE_SIMILARITY_DOC[] =
"Compute homography parameters for a similarity transform.";

static PyMethodDef native_module_methods[] = {
    {"noop",       (PyCFunction)noop,       METH_VARARGS, NATIVE_NOOP_DOC},
    {"similarity", (PyCFunction)similarity, METH_VARARGS, NATIVE_SIMILARITY_DOC},
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

