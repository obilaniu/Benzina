/* Includes */
#include <Python.h>        /* Because of "reasons", the Python header must be first. */
#include "structmember.h"
#include <dlfcn.h>
#include <stdint.h>
#include "./native_nvdecodedataloaderitercore.h"
#include "./native_nvdecodedataloaderitercorebatchcm.h"



/* Python API Function Definitions */

/**
 * @brief Slot tp_alloc
 * 
 * Pure allocation of memory.
 * 
 * Custom garbage-collector-enabled objects require the use of the special allocator
 * PyObject_GC_New().
 */

static PyObject* NvdecodeDataLoaderIterCore_alloc                   (PyTypeObject* type,
                                                                     Py_ssize_t    nitems){
	(void)nitems;
	return (PyObject*)PyObject_GC_New(NvdecodeDataLoaderIterCore, type);
}

/**
 * @brief Slot tp_dealloc
 * 
 * Ensure object is finalized (unless resurrected), then deallocates it.
 * 
 * See extensive commentary inside this function regarding finalization and
 * circular references.
 */

static void      NvdecodeDataLoaderIterCore_dealloc                 (NvdecodeDataLoaderIterCore* self){
	/**
	 * The object now has a reference count of 0, and the intent is to destroy it.
	 * Its finalizer may or may not have been invoked already; If the finalizer
	 * has already been invoked, some of its fields may already have been cleared
	 * as well.
	 * 
	 * Call the tp_finalizer slot with the wrapper PyObject_CallFinalizerFromDealloc().
	 * The execution of the finalizer can lead to potentially arbitrary Python code
	 * being executed, and can even cause the resurrection of self. The wrapper
	 * therefore temporarily bumps the reference count back to 1, executes the
	 * finalizer and then brings the reference count back down again. If it is not
	 * again 0, resurrection has happened.
	 * 
	 * If the finalizer had already been called, this is a no-op.
	 * If the finalizer hadn't been called, it will be called now. If it resurrects
	 * this object, we abort the deallocation and return.
	 */
	
	if(PyObject_CallFinalizerFromDealloc((PyObject*)self) != 0){
		return;
	}
	
	/**
	 * This object has now been finalized exactly once, and its reference count is 0.
	 * It is not in any CI (cyclic isolate) reference loop, since any that did exist
	 * that included this object were broken by the tp_clear slot of one of the CI
	 * objects (possibly even this object's!), thus allowing the reference count of
	 * this object to drop down to 0 and enter this function. This object is doomed.
	 * 
	 * We now have full authority to destroy this object. We proceed by clearing any
	 * remaining direct references to PyObject* fields, then all C handles/pointers/
	 * values, and lastly we free this object itself.
	 * 
	 * For the clearing of the PyObject* fields, we reuse our tp_clear slot
	 * implementation. This is safe in this particular type because this type's
	 * tp_clear slot was designed to be idempotent. Indeed, there is no good reason
	 * why tp_clear shouldn't be idempotent.
	 * 
	 * This function (the tp_dealloc slot) cannot be re-entered as a result of the
	 * use of the tp_clear slot and/or Py_CLEAR() calls, because Py_CLEAR() first
	 * makes a temporary copy of the reference, NULLs out the original source, and
	 * only then decrements the reference count on the reference. Therefore, if the
	 * breaking of the reference cycle began at this object's referrer (i.e., this
	 * object's referrer is the one who invoked Py_CLEAR() on us), the chain of
	 * destructors *cannot* eventually loop back here, since our referrer's
	 * reference to us is already NULL were we to check, and the argument to this
	 * function probably came from a temporary variable.
	 */
	
	//PyObject_GC_UnTrack(self);
	Py_TYPE(self)->tp_clear((PyObject*)self);
	
	/**
	 * The tp_clear() is only required to clear Python objects that contribute
	 * to circular references. In our design we also deallocate all other
	 * Python objects as well. But we also have non-Python, C handles to release.
	 * We do so here.
	 * 
	 * In particular, we release the context object self->ctx, and dlclose()
	 * the plugin handle self->pluginHandle. This may lead to the plugin shared
	 * library potentially being unloaded, as it is reference-counted. When a
	 * shared library is unloaded, any symbols dlsym()'ed from it are
	 * invalidated; For instance, self->v. We therefore erase self->v
	 * immediately before dlclose().
	 */
	
	self->v->release(self->ctx);
	self->v            = NULL;
	dlclose(self->pluginHandle);
	self->pluginHandle = NULL;
	self->ctx          = NULL;
	
	/**
	 * The object is now completely empty. Its memory may now be deallocated and
	 * returned to the system.
	 */
	
	Py_TYPE(self)->tp_free(self);
}

/**
 * @brief Slot tp_traverse
 * 
 * Visit but do not clear all objects contained by this iterator.
 */

static int       NvdecodeDataLoaderIterCore_traverse                (NvdecodeDataLoaderIterCore* self,
                                                                     visitproc                   visit,
                                                                     void*                       arg){
	uint64_t pushes=0, pulls=0, i;
	void*    token = NULL;
	
	self->v->getNumPulls (self->ctx, &pulls);
	self->v->getNumPushes(self->ctx, &pushes);
	
	for(i=pulls;i<pushes;i++){
		if(self->v->peekToken(self->ctx, i, 0, (const void**)&token) == 0){
			if(token){
				Py_VISIT(token);
			}
		}
	}
	Py_VISIT(self->datasetCore);
	Py_VISIT(self->bufferObj);
	
	return 0;
}

/**
 * @brief Slot tp_clear
 * 
 * Clear all objects contained by this iterator.
 */

static int       NvdecodeDataLoaderIterCore_clear                   (NvdecodeDataLoaderIterCore* self){
	uint64_t pushes=0, pulls=0, i;
	void*    token = NULL;
	
	self->v->stopHelpers (self->ctx);
	self->v->getNumPulls (self->ctx, &pulls);
	self->v->getNumPushes(self->ctx, &pushes);
	
	for(i=pulls;i<pushes;i++){
		if(self->v->peekToken(self->ctx, i, 1, (const void**)&token) == 0){
			if(token){
				Py_CLEAR(token);
			}
		}
	}
	Py_CLEAR(self->datasetCore);
	Py_CLEAR(self->bufferObj);
	
	return 0;
}

/**
 * @brief Slot tp_finalize
 * 
 * The finalizer receives this object while it and all of its referent objects
 * are sane (not partially destructed, invalid zombie objects), and should
 * leave this object in a sane state as well. Ideally, it would be more "closed",
 * empty or deinitialized than before.
 * 
 * In the process of doing so, the finalizer may (and even should) contribute to
 * breaking reference cycles, potentially leading to this object's own destruction.
 * It can do that by deleting attributes and/or replacing them with something like
 * None.
 */

static void      NvdecodeDataLoaderIterCore_finalize                (NvdecodeDataLoaderIterCore* self){
	uint64_t pushes=0, pulls=0, i;
	void*    token = NULL;
	PyObject* error_type, *error_value, *error_traceback;
	
	PyErr_Fetch(&error_type, &error_value, &error_traceback);
	
	/**
	 * We empty the pipeline, since we filled it with Python objects.
	 * Because we're in a finalizer, we're entitled to believe that no-one
	 * wants to collect them, so we destroy them.
	 */
	
	self->v->stopHelpers (self->ctx);
	self->v->getNumPulls (self->ctx, &pulls);
	self->v->getNumPushes(self->ctx, &pushes);
	for(i=pulls;i<pushes;i++){
		if(self->v->peekToken(self->ctx, i, 1, (const void**)&token) == 0){
			if(token){
				Py_CLEAR(token);
			}
		}
	}
	
	/**
	 * Then, set our attributes to None.
	 */
	
	#define Py_CLEAR2NONE(x)                \
	    do{                                 \
	        PyObject* tmp = (PyObject*)(x); \
	        Py_INCREF(Py_None);             \
	        (x) = (void*)Py_None;           \
	        Py_CLEAR(tmp);                  \
	    }while(0)
	Py_CLEAR2NONE(self->datasetCore);
	Py_CLEAR2NONE(self->bufferObj);
	#undef Py_CLEAR2NONE
	
	PyErr_Restore(error_type, error_value, error_traceback);
}

/**
 * @brief Slot tp_new
 * 
 * Create a new object. As an early check, attempts to load the corresponding
 * plugin shared library. If this fails, raise an exception. Else, proceed with
 * object allocation.
 */

static PyObject* NvdecodeDataLoaderIterCore_new                     (PyTypeObject* type,
                                                                     PyObject*     args,
                                                                     PyObject*     kwargs){
	(void)args;
	(void)kwargs;
	
	void* pluginHandle, *v;
	
	pluginHandle = dlopen("libbenzina-plugin-nvdecode.so", RTLD_LAZY);
	if(!pluginHandle){
		PyErr_SetString(PyExc_ImportError, "Could not load libbenzina-plugin-nvdecode.so!");
		return NULL;
	}
	v = dlsym(pluginHandle, "VTABLE");
	if(!v){
		dlclose(pluginHandle);
		PyErr_SetString(PyExc_ImportError, "Incompatible libbenzina-plugin-nvdecode.so found!");
		return NULL;
	}
	
	NvdecodeDataLoaderIterCore* self = (NvdecodeDataLoaderIterCore*)type->tp_alloc(type, 0);
	
	if(self){
		self->pluginHandle = pluginHandle;
		self->v            = v;
		self->datasetCore  = NULL;
		self->ctx          = NULL;
		//PyObject_GC_Track(self);
	}else{
		dlclose(pluginHandle);
	}
	
	return (PyObject*)self;
}

/**
 * @brief Slot tp_init
 * 
 * Initialize the object.
 */

static int       NvdecodeDataLoaderIterCore_init                    (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
	void*              ctx            = NULL;
	DatasetCore*       datasetCore    = NULL;
	const char*        deviceId       = NULL;
	PyObject*          bufferObj      = NULL;
	unsigned long long bufferPtr      = 0;
	unsigned long long batchSize      = 256;
	unsigned long long multibuffering = 3;
	unsigned long long outputHeight   = 256;
	unsigned long long outputWidth    = 256;
	
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

static PyObject* NvdecodeDataLoaderIterCore_getPushes               (NvdecodeDataLoaderIterCore* self, void *closure){
	uint64_t i = 0;
	if(self->ctx){self->v->getNumPushes     (self->ctx, &i);}
	return PyLong_FromUnsignedLongLong(i);
}
static PyObject* NvdecodeDataLoaderIterCore_getPulls                (NvdecodeDataLoaderIterCore* self, void *closure){
	uint64_t i = 0;
	if(self->ctx){self->v->getNumPulls      (self->ctx, &i);}
	return PyLong_FromUnsignedLongLong(i);
}
static PyObject* NvdecodeDataLoaderIterCore_getMultibuffering       (NvdecodeDataLoaderIterCore* self, void *closure){
	uint64_t i = 0;
	if(self->ctx){self->v->getMultibuffering(self->ctx, &i);}
	return PyLong_FromUnsignedLongLong(i);
}


/**
 * METHODS
 */

static PyObject* NvdecodeDataLoaderIterCore_batch                   (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
	PyObject* token = NULL, *ret;
	
	static char *kwargsList[] = {"token", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwargsList,
	                                &token)){
		return NULL;
	}
	
	if(token == NULL){
		token = Py_None;
	}
	
	ret = PyObject_CallFunction((PyObject*)&NvdecodeDataLoaderIterCoreBatchCMType,
	                            "OO", self, token);
	return ret;
}
static PyObject* NvdecodeDataLoaderIterCore_defineBatch             (NvdecodeDataLoaderIterCore* self){
	if(self->v->defineBatch(self->ctx) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in defineBatch()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* NvdecodeDataLoaderIterCore_submitBatch             (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_waitBatch               (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_defineSample            (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
	unsigned long long  datasetIndex        = -1;
	unsigned long long  devicePtr           = 0;
	PyObject*           pySample            = NULL;
	PyObject*           pyLocation          = NULL;
	PyObject*           pyConfigLocation    = NULL;
	uint64_t            location[2]         = {0, 0};
	uint64_t            configLocation[2]   = {0, 0};
	
	static char *kwargsList[] = {"datasetIndex", "dstPtr", "sample", "location", "config_location", NULL};
	
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "KKOOO", kwargsList,
	                                &datasetIndex, &devicePtr, &pySample, &pyLocation, &pyConfigLocation)){
		PyErr_SetString(PyExc_RuntimeError,
		                "Could not parse arguments!");
		return NULL;
	}
	
	if(pySample != NULL && !PyMemoryView_Check(pySample)){
		Py_DECREF(pySample);
		pySample = NULL;
	}

	if(pyLocation != NULL && !PyTuple_Check(pyLocation) &&
	   !PyArg_ParseTuple(pyLocation, "kk", &location[0], &location[1])){
        Py_DECREF(pyLocation);
        pyLocation = NULL;
	}

	if(pyConfigLocation != NULL && !PyTuple_Check(pyConfigLocation) &&
	   !PyArg_ParseTuple(pyConfigLocation, "kk", &configLocation[0], &configLocation[1])){
        Py_DECREF(pyConfigLocation);
        pyConfigLocation = NULL;
	}
	
	if(pySample == NULL || pyLocation == NULL || pyConfigLocation == NULL ||
	   self->v->defineSample(self->ctx, datasetIndex, (void*)devicePtr, PyMemoryView_GET_BUFFER(pySample)->buf,
	                         location, configLocation) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in defineSample()!");
		return NULL;
	}
	
	// Ownership of sample is kept in benzina.torch.dataloader._DataLoaderIter
	// Py_DECREF(pySample);
	Py_DECREF(pyLocation);
	Py_DECREF(pyConfigLocation);
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* NvdecodeDataLoaderIterCore_submitSample            (NvdecodeDataLoaderIterCore* self){
	if(self->v->submitSample(self->ctx) != 0){
		PyErr_SetString(PyExc_RuntimeError,
		                "Error in submitSample()!");
		return NULL;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}
static PyObject* NvdecodeDataLoaderIterCore_setHomography           (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_setBias                 (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_setScale                (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_setOOBColor             (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_selectColorMatrix       (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_setDefaultBias          (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_setDefaultScale         (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_setDefaultOOBColor      (NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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
static PyObject* NvdecodeDataLoaderIterCore_selectDefaultColorMatrix(NvdecodeDataLoaderIterCore* self,
                                                                     PyObject*                   args,
                                                                     PyObject*                   kwargs){
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

static PyGetSetDef NvdecodeDataLoaderIterCore_getsetters[] = {
    {"pushes",                   (getter)NvdecodeDataLoaderIterCore_getPushes,         0, "Number of batches pushed into the core",                NULL},
    {"pulls",                    (getter)NvdecodeDataLoaderIterCore_getPulls,          0, "Number of batches pulled out of the core",              NULL},
    {"multibuffering",           (getter)NvdecodeDataLoaderIterCore_getMultibuffering, 0, "Maximum number of batches outstanding within the core", NULL},
    {NULL}  /* Sentinel */
};

/**
 * Methods table.
 */

static PyMethodDef NvdecodeDataLoaderIterCore_methods[] = {
    {"batch",                    (PyCFunction)NvdecodeDataLoaderIterCore_batch,                    METH_VARARGS|METH_KEYWORDS, "Create a new batch definition context."},
    {"defineBatch",              (PyCFunction)NvdecodeDataLoaderIterCore_defineBatch,              METH_NOARGS,                "Begin defining a new batch of samples."},
    {"submitBatch",              (PyCFunction)NvdecodeDataLoaderIterCore_submitBatch,              METH_VARARGS|METH_KEYWORDS, "Submit a new batch of samples."},
    {"waitBatch",                (PyCFunction)NvdecodeDataLoaderIterCore_waitBatch,                METH_VARARGS|METH_KEYWORDS, "Wait for a batch of samples to become complete."},
    {"defineSample",             (PyCFunction)NvdecodeDataLoaderIterCore_defineSample,             METH_VARARGS|METH_KEYWORDS, "Begin defining a new sample within a batch."},
    {"submitSample",             (PyCFunction)NvdecodeDataLoaderIterCore_submitSample,             METH_NOARGS,                "Submit a new sample within a batch."},
    {"setHomography",            (PyCFunction)NvdecodeDataLoaderIterCore_setHomography,            METH_VARARGS|METH_KEYWORDS, "Set homography for this job."},
    {"setBias",                  (PyCFunction)NvdecodeDataLoaderIterCore_setBias,                  METH_VARARGS|METH_KEYWORDS, "Set bias for this job."},
    {"setScale",                 (PyCFunction)NvdecodeDataLoaderIterCore_setScale,                 METH_VARARGS|METH_KEYWORDS, "Set scale for this job."},
    {"setOOBColor",              (PyCFunction)NvdecodeDataLoaderIterCore_setOOBColor,              METH_VARARGS|METH_KEYWORDS, "Set out-of-bounds color for this job."},
    {"selectColorMatrix",        (PyCFunction)NvdecodeDataLoaderIterCore_selectColorMatrix,        METH_VARARGS|METH_KEYWORDS, "Select color matrix for this job."},
    {"setDefaultBias",           (PyCFunction)NvdecodeDataLoaderIterCore_setDefaultBias,           METH_VARARGS|METH_KEYWORDS, "Set default bias."},
    {"setDefaultScale",          (PyCFunction)NvdecodeDataLoaderIterCore_setDefaultScale,          METH_VARARGS|METH_KEYWORDS, "Set default scale."},
    {"setDefaultOOBColor",       (PyCFunction)NvdecodeDataLoaderIterCore_setDefaultOOBColor,       METH_VARARGS|METH_KEYWORDS, "Set default out-of-bounds color."},
    {"selectDefaultColorMatrix", (PyCFunction)NvdecodeDataLoaderIterCore_selectDefaultColorMatrix, METH_VARARGS|METH_KEYWORDS, "Select default color matrix."},
    {NULL}  /* Sentinel */
};

static PyTypeObject NvdecodeDataLoaderIterCoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "benzina.native.NvdecodeDataLoaderIterCore",       /* tp_name */
    sizeof(NvdecodeDataLoaderIterCore),                /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)NvdecodeDataLoaderIterCore_dealloc,    /* tp_dealloc */
    0,                                                 /* tp_print */
    0,                                                 /* tp_getattr */
    0,                                                 /* tp_setattr */
    0,                                                 /* tp_reserved */
    0,                                                 /* tp_repr */
    0,                                                 /* tp_as_number */
    0,                                                 /* tp_as_sequence */
    0,                                                 /* tp_as_mapping */
    0,                                                 /* tp_hash  */
    0,                                                 /* tp_call */
    0,                                                 /* tp_str */
    0,                                                 /* tp_getattro */
    0,                                                 /* tp_setattro */
    0,                                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_FINALIZE |
        Py_TPFLAGS_HAVE_GC,                            /* tp_flags */
    "NvdecodeDataLoaderIterCore objects",              /* tp_doc */
    (traverseproc)NvdecodeDataLoaderIterCore_traverse, /* tp_traverse */
    (inquiry)NvdecodeDataLoaderIterCore_clear,         /* tp_clear */
    0,                                                 /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    0,                                                 /* tp_iter */
    0,                                                 /* tp_iternext */
    NvdecodeDataLoaderIterCore_methods,                /* tp_methods */
    0,                                                 /* tp_members */
    NvdecodeDataLoaderIterCore_getsetters,             /* tp_getset */
    0,                                                 /* tp_base */
    0,                                                 /* tp_dict */
    0,                                                 /* tp_descr_get */
    0,                                                 /* tp_descr_set */
    0,                                                 /* tp_dictoffset */
    (initproc)NvdecodeDataLoaderIterCore_init,         /* tp_init */
    NvdecodeDataLoaderIterCore_alloc,                  /* tp_alloc */
    NvdecodeDataLoaderIterCore_new,                    /* tp_new */
    0,                                                 /* tp_free */
    0,                                                 /* tp_is_gc */
    0,                                                 /* tp_bases */
    0,                                                 /* tp_mro */
    0,                                                 /* tp_cache */
    0,                                                 /* tp_subclasses */
    0,                                                 /* tp_weaklist */
    0,                                                 /* tp_del */
    0,                                                 /* tp_version_tag */
    (destructor)NvdecodeDataLoaderIterCore_finalize,   /* tp_finalize */
};
