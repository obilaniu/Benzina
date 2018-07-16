/* Includes */
#define  PY_SSIZE_T_CLEAN  /* So we get Py_ssize_t args. */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include "structmember.h"
#include <dlfcn.h>
#include <stdint.h>
#include "native.h"


/* Defines */



/* Python API Function Definitions */

/**
 * @brief Slot tp_dealloc
 */

static void      BenzinaPluginNvdecodeCore_dealloc                 (BenzinaPluginNvdecodeCore* self){
	void* token;
	
	while(self->v->waitToken(self->ctx, &token) == 0){
		Py_CLEAR(token);
	}
	self->v->release(self->ctx);
	
	Py_CLEAR(self->datasetCore);
	Py_CLEAR(self->bufferObj);
	dlclose(self->pluginHandle);
	self->pluginHandle = NULL;
	self->v            = NULL;
	self->ctx          = NULL;
	Py_TYPE(self)->tp_free(self);
}

/**
 * @brief Slot tp_new
 */

static PyObject* BenzinaPluginNvdecodeCore_new                     (PyTypeObject* type,
                                                                    PyObject*     args,
                                                                    PyObject*     kwargs){
	void* pluginHandle, *v;
	
	pluginHandle = dlopen("libbenzina_plugin_nvdecode.so", RTLD_LAZY);
	if(!pluginHandle){
		PyErr_SetString(PyExc_ImportError, "Could not load libbenzina_plugin_nvdecode.so!");
		return NULL;
	}
	v = dlsym(pluginHandle, "VTABLE");
	if(!v){
		dlclose(pluginHandle);
		PyErr_SetString(PyExc_ImportError, "Incompatible libbenzina_plugin_nvdecode.so found!");
		return NULL;
	}
	
	BenzinaPluginNvdecodeCore* self = (BenzinaPluginNvdecodeCore*)type->tp_alloc(type, 0);
	
	if(self){
		self->pluginHandle = pluginHandle;
		self->v            = v;
		self->datasetCore  = NULL;
		self->ctx          = NULL;
	}else{
		dlclose(pluginHandle);
	}
	
	return (PyObject*)self;
}

/**
 * @brief Slot tp_init
 */

static int       BenzinaPluginNvdecodeCore_init                    (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	void*               ctx            = NULL;
	BenzinaDatasetCore* datasetCore    = NULL;
	const char*         deviceId       = "cuda:0";
	PyObject*           bufferObj      = NULL;
	unsigned long long  bufferPtr      = 0;
	unsigned long long  batchSize      = 256;
	unsigned long long  multibuffering = 3;
	unsigned long long  outputHeight   = 256;
	unsigned long long  outputWidth    = 256;
	
	static char *kwargsList[] = {"dataset",
	                             "deviceId",
	                             "bufferObj",
	                             "bufferPtr",
	                             "batchSize",
	                             "multibuffering",
	                             "outputHeight",
	                             "outputWidth",
	                             NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OsOK|KKKK", kwargsList,
	                                &datasetCore,  &deviceId,  &bufferObj,
	                                &bufferPtr,    &batchSize, &multibuffering,
	                                &outputHeight, &outputWidth)){
		return -1;
	}
	
	if(self->v->alloc(&ctx, datasetCore->dataset) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error during creation of decoding context!");
		return -1;
	}
	
	if(self->v->setBuffer(ctx, deviceId, (void*)bufferPtr, multibuffering,
	                      batchSize, outputHeight, outputWidth) != 0){
		self->v->release(ctx);
		PyErr_SetString(PyExc_RuntimeError,
		                "Error during installation of decoding context's target"
		                " buffer and geometry!");
		return -1;
	}
	
	if(self->v->init(ctx) != 0){
		self->v->release(ctx);
		PyErr_SetString(PyExc_RuntimeError,
		                "Failed initializing decoder context!");
		return -1;
	}
	
	Py_INCREF(datasetCore);
	Py_INCREF(bufferObj);
	self->bufferObj   = bufferObj;
	self->datasetCore = datasetCore;
	self->ctx         = ctx;
	return 0;
}

/**
 * GETTER/SETTER
 */

static PyObject* BenzinaPluginNvdecodeCore_getPushes               (BenzinaPluginNvdecodeCore* self, void *closure){
	uint64_t i = 0;
	if(self->ctx){self->v->getNumPushes     (self->ctx, &i);}
	return PyLong_FromUnsignedLongLong(i);
}
static PyObject* BenzinaPluginNvdecodeCore_getPulls                (BenzinaPluginNvdecodeCore* self, void *closure){
	uint64_t i = 0;
	if(self->ctx){self->v->getNumPulls      (self->ctx, &i);}
	return PyLong_FromUnsignedLongLong(i);
}
static PyObject* BenzinaPluginNvdecodeCore_getMultibuffering       (BenzinaPluginNvdecodeCore* self, void *closure){
	uint64_t i = 0;
	if(self->ctx){self->v->getMultibuffering(self->ctx, &i);}
	return PyLong_FromUnsignedLongLong(i);
}


/**
 * METHODS
 */

static PyObject* BenzinaPluginNvdecodeCore_defineBatch             (BenzinaPluginNvdecodeCore* self){
	if(self->v->defineBatch(self->ctx) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in defineBatch()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_submitBatch             (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	void*  token   = NULL;
	
	static char *kwargsList[] = {"token", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwargsList,
	                                &token)){
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not parse arguments!");
		return NULL;
	}
	
	Py_XINCREF(token);
	if(self->v->submitBatch(self->ctx, token) != 0){
		Py_XDECREF(token);
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in submitBatch()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_waitBatch               (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	void*  token   = NULL;
	int    block   = 1;
	double timeout = 0;
	int    ret;
	
	static char *kwargsList[] = {"block", "timeout", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "pd", kwargsList,
	                                &block, &timeout)){
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not parse arguments!");
		return NULL;
	}
	
	Py_BEGIN_ALLOW_THREADS
	ret = self->v->waitBatch(self->ctx, (const void**)&token, block, timeout);
	Py_END_ALLOW_THREADS
	
	switch(ret){
		case ETIMEDOUT:
			PyErr_SetString(PyExc_TimeoutError, "waitBatch() timed out!");
			Py_XDECREF(token);
		return NULL;
		default:
			PyErr_SetString(PyExc_RuntimeError, "Error in pull()!");
			Py_XDECREF(token);
		return NULL;
		case 0:
			if(!token){
				token = Py_None;
				Py_INCREF(token);
			}
		return token;
	}
}
static PyObject* BenzinaPluginNvdecodeCore_defineSample            (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	unsigned long long  datasetIndex   = -1;
	unsigned long long  devicePtr      = 0;
	
	static char *kwargsList[] = {"datasetIndex",
	                             "dstPtr",
	                             NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "KK", kwargsList,
	                                &datasetIndex, &devicePtr)){
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not parse arguments!");
		return NULL;
	}
	
	if(self->v->defineSample(self->ctx, datasetIndex, (void*)devicePtr) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in defineSample()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_submitSample            (BenzinaPluginNvdecodeCore* self){
	if(self->v->submitSample(self->ctx) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in submitSample()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_setHomography           (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	float H[3][3] = {{1,0,0},
	                 {0,1,0},
	                 {0,0,1}};
	
	static char *kwargsList[] = {"H00", "H01", "H02",
	                             "H10", "H11", "H12",
	                             "H20", "H21", "H22",
	                             NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "fffffffff", kwargsList,
	                                &H[0][0], &H[0][1], &H[0][2],
	                                &H[1][0], &H[1][1], &H[1][2],
	                                &H[2][0], &H[2][1], &H[2][2])){
		PyErr_SetString(PyExc_RuntimeError,
		                "Must pass 3x3 floating-point numbers!");
		return NULL;
	}
	
	if(self->v->setHomography(self->ctx, (const float*)H) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in setHomography()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_setBias                 (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	float B[3] = {0,0,0};
	
	static char *kwargsList[] = {"B0", "B1", "B2", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "fff", kwargsList,
	                                &B[0], &B[1], &B[2])){
		PyErr_SetString(PyExc_RuntimeError,
		                "Must pass 3 floating-point numbers!");
		return NULL;
	}
	
	if(self->v->setBias(self->ctx, B) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in setBias()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_setScale                (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	float S[3] = {0,0,0};
	
	static char *kwargsList[] = {"S0", "S1", "S2", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "fff", kwargsList,
	                                &S[0], &S[1], &S[2])){
		PyErr_SetString(PyExc_RuntimeError,
		                "Must pass 3 floating-point numbers!");
		return NULL;
	}
	
	if(self->v->setScale(self->ctx, S) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in setScale()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_setOOBColor             (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	float OOB[3] = {0,0,0};
	
	static char *kwargsList[] = {"OOB0", "OOB1", "OOB2", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "fff", kwargsList,
	                                &OOB[0], &OOB[1], &OOB[2])){
		PyErr_SetString(PyExc_RuntimeError,
		                "Must pass 3 floating-point numbers!");
		return NULL;
	}
	
	if(self->v->setOOBColor(self->ctx, OOB) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in setOOBColor()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_selectColorMatrix       (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	unsigned long long colorMatrix = 0;
	
	static char *kwargsList[] = {"colorMatrix", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "K", kwargsList,
	                                &colorMatrix)){
		PyErr_SetString(PyExc_RuntimeError,
		                "Must pass an integer number identifying a color matrix!");
		return NULL;
	}
	
	if(self->v->selectColorMatrix(self->ctx, colorMatrix) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in selectColorMatrix()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_setDefaultBias          (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	float B[3] = {0,0,0};
	
	static char *kwargsList[] = {"B0", "B1", "B2", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "fff", kwargsList,
	                                &B[0], &B[1], &B[2])){
		PyErr_SetString(PyExc_RuntimeError,
		                "Must pass 3 floating-point numbers!");
		return NULL;
	}
	
	if(self->v->setDefaultBias(self->ctx, B) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in setDefaultBias()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_setDefaultScale         (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	float S[3] = {0,0,0};
	
	static char *kwargsList[] = {"S0", "S1", "S2", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "fff", kwargsList,
	                                &S[0], &S[1], &S[2])){
		PyErr_SetString(PyExc_RuntimeError,
		                "Must pass 3 floating-point numbers!");
		return NULL;
	}
	
	if(self->v->setDefaultScale(self->ctx, S) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in setDefaultScale()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_setDefaultOOBColor      (BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	float OOB[3] = {0,0,0};
	
	static char *kwargsList[] = {"OOB0", "OOB1", "OOB2", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "fff", kwargsList,
	                                &OOB[0], &OOB[1], &OOB[2])){
		PyErr_SetString(PyExc_RuntimeError,
		                "Must pass 3 floating-point numbers!");
		return NULL;
	}
	
	if(self->v->setDefaultOOBColor(self->ctx, OOB) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in setDefaultOOBColor()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* BenzinaPluginNvdecodeCore_selectDefaultColorMatrix(BenzinaPluginNvdecodeCore* self,
                                                                    PyObject*                  args,
                                                                    PyObject*                  kwargs){
	unsigned long long colorMatrix = 0;
	
	static char *kwargsList[] = {"colorMatrix", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "K", kwargsList,
	                                &colorMatrix)){
		PyErr_SetString(PyExc_RuntimeError,
		                "Must pass an integer number identifying a color matrix!");
		return NULL;
	}
	
	if(self->v->selectDefaultColorMatrix(self->ctx, colorMatrix) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in selectDefaultColorMatrix()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}

/**
 * Getter/Setter table.
 */

static PyGetSetDef BenzinaPluginNvdecodeCore_getsetters[] = {
    {"pushes",                   (getter)BenzinaPluginNvdecodeCore_getPushes,         0, "Number of batches pushed into the core",                NULL},
    {"pulls",                    (getter)BenzinaPluginNvdecodeCore_getPulls,          0, "Number of batches pulled out of the core",              NULL},
    {"multibuffering",           (getter)BenzinaPluginNvdecodeCore_getMultibuffering, 0, "Maximum number of batches outstanding within the core", NULL},
    {NULL}  /* Sentinel */
};

/**
 * Methods table.
 */

static PyMethodDef BenzinaPluginNvdecodeCore_methods[] = {
    {"defineBatch",              (PyCFunction)BenzinaPluginNvdecodeCore_defineBatch,              METH_NOARGS,                "Begin defining a new batch of samples."},
    {"submitBatch",              (PyCFunction)BenzinaPluginNvdecodeCore_submitBatch,              METH_VARARGS|METH_KEYWORDS, "Submit a new batch of samples."},
    {"waitBatch",                (PyCFunction)BenzinaPluginNvdecodeCore_waitBatch,                METH_VARARGS|METH_KEYWORDS, "Wait for a batch of samples to become complete."},
    {"defineSample",             (PyCFunction)BenzinaPluginNvdecodeCore_defineSample,             METH_VARARGS|METH_KEYWORDS, "Begin defining a new sample within a batch."},
    {"submitSample",             (PyCFunction)BenzinaPluginNvdecodeCore_submitSample,             METH_NOARGS,                "Submit a new sample within a batch."},
    {"setHomography",            (PyCFunction)BenzinaPluginNvdecodeCore_setHomography,            METH_VARARGS|METH_KEYWORDS, "Set homography for this job."},
    {"setBias",                  (PyCFunction)BenzinaPluginNvdecodeCore_setBias,                  METH_VARARGS|METH_KEYWORDS, "Set bias for this job."},
    {"setScale",                 (PyCFunction)BenzinaPluginNvdecodeCore_setScale,                 METH_VARARGS|METH_KEYWORDS, "Set scale for this job."},
    {"setOOBColor",              (PyCFunction)BenzinaPluginNvdecodeCore_setOOBColor,              METH_VARARGS|METH_KEYWORDS, "Set out-of-bounds color for this job."},
    {"selectColorMatrix",        (PyCFunction)BenzinaPluginNvdecodeCore_selectColorMatrix,        METH_VARARGS|METH_KEYWORDS, "Select color matrix for this job."},
    {"setDefaultBias",           (PyCFunction)BenzinaPluginNvdecodeCore_setDefaultBias,           METH_VARARGS|METH_KEYWORDS, "Set default bias."},
    {"setDefaultScale",          (PyCFunction)BenzinaPluginNvdecodeCore_setDefaultScale,          METH_VARARGS|METH_KEYWORDS, "Set default scale."},
    {"setDefaultOOBColor",       (PyCFunction)BenzinaPluginNvdecodeCore_setDefaultOOBColor,       METH_VARARGS|METH_KEYWORDS, "Set default out-of-bounds color."},
    {"selectDefaultColorMatrix", (PyCFunction)BenzinaPluginNvdecodeCore_selectDefaultColorMatrix, METH_VARARGS|METH_KEYWORDS, "Select default color matrix."},
	{NULL}  /* Sentinel */
};

static PyTypeObject BenzinaPluginNvdecodeCoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "native.BenzinaPluginNvdecodeCore",             /* tp_name */
    sizeof(BenzinaPluginNvdecodeCore),              /* tp_basicsize */
    0,                                              /* tp_itemsize */
    (destructor)BenzinaPluginNvdecodeCore_dealloc,  /* tp_dealloc */
    0,                                              /* tp_print */
    0,                                              /* tp_getattr */
    0,                                              /* tp_setattr */
    0,                                              /* tp_reserved */
    0,                                              /* tp_repr */
    0,                                              /* tp_as_number */
    0,                                              /* tp_as_sequence */
    0,                                              /* tp_as_mapping */
    0,                                              /* tp_hash  */
    0,                                              /* tp_call */
    0,                                              /* tp_str */
    0,                                              /* tp_getattro */
    0,                                              /* tp_setattro */
    0,                                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,         /* tp_flags */
    "BenzinaPluginNvdecodeCore objects",            /* tp_doc */
    0,                                              /* tp_traverse */
    0,                                              /* tp_clear */
    0,                                              /* tp_richcompare */
    0,                                              /* tp_weaklistoffset */
    0,                                              /* tp_iter */
    0,                                              /* tp_iternext */
    BenzinaPluginNvdecodeCore_methods,              /* tp_methods */
    0,                                              /* tp_members */
    BenzinaPluginNvdecodeCore_getsetters,           /* tp_getset */
    0,                                              /* tp_base */
    0,                                              /* tp_dict */
    0,                                              /* tp_descr_get */
    0,                                              /* tp_descr_set */
    0,                                              /* tp_dictoffset */
    (initproc)BenzinaPluginNvdecodeCore_init,       /* tp_init */
    0,                                              /* tp_alloc */
    BenzinaPluginNvdecodeCore_new,                  /* tp_new */
};
