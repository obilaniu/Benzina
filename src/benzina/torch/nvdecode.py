# -*- coding: utf-8 -*-
import benzina.native
import gc
import numpy                           as np
import os, sys
import torch
import torch.utils.data

from   torch.utils.data.dataloader import default_collate
from   contextlib                  import suppress



class NvdecodeDataLoader(torch.utils.data.DataLoader):
	"""
	Loads images from a benzina.torch.dataset.Dataset. Encapsulates a sampler
	and data processing transformations.

	Arguments
	---------
	dataset (benzina.torch.dataset.Dataset):
		dataset from which to load the data.
	batch_size : int, optional
		how many samples per batch to load (default: ``1``).
	shuffle (bool, optional): set to ``True`` to have the data reshuffled at every
		epoch (default: ``False``).
	sampler (Sampler, optional): defines the strategy to draw samples from the
		dataset. If specified, ``shuffle`` must be False.
	batch_sampler (Sampler, optional): like sampler, but returns a batch of indices
		at a time. Mutually exclusive with :attr:`batch_size`, :attr:`shuffle`,
		:attr:`sampler`, and :attr:`drop_last`.
	collate_fn (callable, optional): merges a list of samples to form a mini-batch.
	drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
		if the dataset size is not divisible by the batch size. If ``False`` and
		the size of dataset is not divisible by the batch size, then the last batch
		will be smaller. (default: ``False``)
	timeout (numeric, optional): if positive, the timeout value for collecting a
		batch. Should always be non-negative. (default: ``0``)
	shape (int or tuple of ints, optional): set the shape of the samples. Note
		that this does not imply a resize of the image but merely set the shape
		of the tensor in which the data will be copied.
	multibuffering (int, optional): set the size of the multibuffering buffer.
		(default: ``3``).
	seed (int, optional): set the seed for the random transformations.
	warp_transform (NvdecodeWarpTransform or iterable of float, optional):
		set the warp transformation or use as the arguments to initialize a
		NvdecodeWarpTransform.
	oob_transform (NvdecodeOOBTransform or float or iterable of float, optional):
		set the out of bounds transformation. Values of a pixel's channels when
		it is out of bound. Note that this transformation is not affected by
		:attr:`scale_transform` or :attr:`bias_transform`.
	color_transform (NvdecodeColorTransform or int, optional): set the color
		transformation.
	scale_transform (NvdecodeScaleTransform or float or iterable of float, optional):
		set the scale transformation. Values to multiply a pixel's channels with.
		Note that this transformation is applied after :attr:`bias_transform`.
	bias_transform (NvdecodeBiasTransform or float, optional):
		set the bias transformation. Values to substract a pixel's channels with.
	"""
	def __init__(self,
	             dataset,
	             batch_size      = 1,
	             shuffle         = False,
	             sampler         = None,
	             batch_sampler   = None,
	             collate_fn      = default_collate,
	             drop_last       = False,
	             timeout         = 0,
	             shape           = None,
	             device_id       = None,
	             multibuffering  = 3,
	             seed            = None,
	             warp_transform  = None,
	             oob_transform   = None,
	             color_transform = None,
	             scale_transform = None,
	             bias_transform  = None):
		super().__init__(dataset,
		                 batch_size     = batch_size,
		                 shuffle        = shuffle,
		                 sampler        = sampler,
		                 batch_sampler  = batch_sampler,
		                 num_workers    = 0,
		                 collate_fn     = collate_fn,
		                 pin_memory     = True,
		                 drop_last      = drop_last,
		                 timeout        = float(timeout),
		                 worker_init_fn = None)
		
		if   shape is None:
			shape = dataset.shape
		elif isinstance(shape, int):
			shape = (shape, shape)
		
		if seed is None:
			seed = torch.randint(low    = 0,
			                     high   = 2**32,
			                     size   = (),
			                     dtype  = torch.int64,
			                     device = "cpu")
			seed = int(seed)
		
		if not isinstance(warp_transform,  NvdecodeWarpTransform):
			warp_transform  = NvdecodeConstantWarpTransform (warp_transform)
		if not isinstance(color_transform, NvdecodeColorTransform):
			color_transform = NvdecodeConstantColorTransform(color_transform)
		if not isinstance(oob_transform,   NvdecodeOOBTransform):
			oob_transform   = NvdecodeConstantOOBTransform  (oob_transform)
		if not isinstance(scale_transform, NvdecodeScaleTransform):
			scale_transform = NvdecodeConstantScaleTransform(scale_transform)
		if not isinstance(bias_transform,  NvdecodeBiasTransform):
			bias_transform  = NvdecodeConstantBiasTransform (bias_transform)
		
		self.device_id       = device_id
		self.multibuffering  = multibuffering
		self.shape           = shape
		self.RNG             = np.random.RandomState(seed)
		self.warp_transform  = warp_transform
		self.color_transform = color_transform
		self.oob_transform   = oob_transform
		self.scale_transform = scale_transform
		self.bias_transform  = bias_transform
	
	def __iter__(self):
		return NvdecodeDataLoaderIter(self)


class NvdecodeDataLoaderIter:
	def __init__(self, loader):
		assert(loader.multibuffering >= 1)
		self.length          = len(loader)
		self.dataset         = loader.dataset
		self.batch_size      = loader.batch_size
		self.batch_iter      = iter(loader.batch_sampler)
		self.multibuffering  = loader.multibuffering
		self.shape           = loader.shape
		self.collate_fn      = loader.collate_fn
		self.drop_last       = loader.drop_last
		self.timeout         = loader.timeout
		if   loader.device_id is None or loader.device_id == "cuda":
			self.device_id = torch.device(torch.cuda.current_device())
		elif isinstance(loader.device_id, (str, int)):
			self.device_id = torch.device(loader.device_id)
		else:
			self.device_id = loader.device_id
		self.RNG             = np.random.RandomState(loader.RNG.randint(2**32))
		self.warp_transform  = loader.warp_transform
		self.color_transform = loader.color_transform
		self.oob_transform   = loader.oob_transform
		self.scale_transform = loader.scale_transform
		self.bias_transform  = loader.bias_transform
		self.multibuffer     = None
		self.core            = None
		self.stop_iteration  = None
	
	def __del__(self):
		"""
		Destroy the iterator and all its resources.
		
		Because extraneous and circular references can keep the large GPU
		multibuffer tensor allocated indefinitely, we:
		
		  1. Forcibly destroy all our members, thereby losing all of the
		     iterator's possible references to the multibuffer and the iterator
		     core. Tensor deallocations may or may not happen at this moment.
		  2. Invoke the garbage collector, which is capable of identifying
		     cyclic trash and removing it. The iterator core object supports
		     garbage collection and is capable of breaking all reference cycles
		     involving it.
		  3. Empty the PyTorch CUDA cache, returning the CUDA memory buffers to
		     the allocation pool.
		
		Because data loaders are not intended to be created extremely often,
		the extra cycles spent here doing this are worth it.
		"""
		
		del self.__dict__
		self.garbage_collect()
	
	def __iter__(self):
		return self
	
	def __len__(self):
		return self.length
	
	def __next__(self):
		if self.stop_iteration is not None:
			raise self.stop_iteration
		try:
			if self.core_needs_init():
				self.pull_first_indices()
				self.init_core()
				self.push_first_indices()
				with suppress(StopIteration):
					self.fill_core()
			else:
				with suppress(StopIteration):
					self.fill_one_batch()
			return self.pull()
		except StopIteration as si:
			self.stop_iteration = si
			self.garbage_collect()
			raise self.stop_iteration
	
	def core_needs_init(self):
		return self.core is None
	
	def pull_first_indices(self):
		self.first_indices = next(self.batch_iter)
		
	def init_core(self):
		"""
		Initialize the iterator core.
		
		From the first batch drawn from the sample iterator, we know the
		maximum batch size. We allocate a multibuffer large enough to
		containing self.multibuffering batches of the maximum size.
		
		Before we do so, however, we trigger garbage collection and empty
		the tensor cache, in an attempt to ensure circular references
		keeping previous large multibuffers alive have been destroyed.
		"""
		
		self.garbage_collect()
		self.check_or_set_batch_size(self.first_indices)
		self.multibuffer = torch.zeros([self.multibuffering,
		                                self.batch_size,
		                                3,
		                                self.shape[0],
		                                self.shape[1]],
		                               dtype  = torch.float32,
		                               device = self.device_id)
		self.core        = benzina.native.NvdecodeDataLoaderIterCore(
		                       self.dataset._core,
		                       str(self.device_id),
		                       self.multibuffer,
		                       self.multibuffer.data_ptr(),
		                       self.batch_size,
		                       self.multibuffering,
		                       self.shape[0],
		                       self.shape[1],
		                   )
	
	def push_first_indices(self):
		self.push(self.__dict__.pop("first_indices"))
	
	def fill_core(self):
		while self.core.pushes < self.core.multibuffering:
			self.fill_one_batch()
	
	def push(self, indices):
		self.check_or_set_batch_size(indices)
		buffer  = self.multibuffer[self.core.pushes % self.core.multibuffering][:len(indices)]
		indices = [int(i)                    for i in indices]
		ptrs    = [int(buffer[n].data_ptr()) for n in range(len(indices))]
		auxd    = [self.dataset[i]           for i in indices]
		token   = (buffer, *self.collate_fn(auxd))
		
		with self.core.batch(token) as batch:
			for i,ptr in zip(indices, ptrs):
				with batch.sample(i, ptr):
					self.core.setHomography    (*self.warp_transform (self, i))
					self.core.selectColorMatrix(*self.color_transform(self, i))
					self.core.setBias          (*self.bias_transform (self, i))
					self.core.setScale         (*self.scale_transform(self, i))
					self.core.setOOBColor      (*self.oob_transform  (self, i))
	
	def pull(self):
		if self.core.pulls >= self.core.pushes:
			raise StopIteration
		return self.core.waitBatch(block=True, timeout=self.timeout)
	
	def fill_one_batch(self):
		self.push(next(self.batch_iter))
	
	def check_or_set_batch_size(self, indices):
		iter_batch_size = len(indices)
		
		if   self.batch_size is None:
			self.batch_size = iter_batch_size
		elif self.batch_size < iter_batch_size:
			raise RuntimeError("Batch size expected to be {}, but iterator returned larger batch size {}!"
			                   .format(self.batch_size, iter_batch_size))
		elif self.batch_size > iter_batch_size:
			if self.drop_last:
				raise StopIteration
	
	def garbage_collect(self):
		self.core           = None
		self.multibuffer    = None
		gc.collect()
		torch.cuda.empty_cache()


class NvdecodeWarpTransform:
	"""
	Interface class that represents a warp transformation (rotation, offset,
	scale, skew).
	"""
	def __init__(self, *args, **kwargs):
		pass
	"""
	__call__ needs to be implemented in subclasses

	Arguments
	---------

	Returns
	-------
	out (tuple):
		a flatten 3 x 3 matrix returned in a tuple of numerics. It should be in
		row-major order.
	"""
	def __call__(self, dataloaderiter, i):
		return (1.0, 0.0, 0.0,
		        0.0, 1.0, 0.0,
		        0.0, 0.0, 1.0)
class NvdecodeOOBTransform:
	"""
	Interface class that represents an out of bounds transformation.
	"""
	def __init__(self, *args, **kwargs):
		pass
	"""
	__call__ needs to be implemented in subclasses

	Arguments
	---------

	Returns
	-------
	out (tuple):
		a tuple of numerics containing the RGB color to use when no data is available.
		It should be in RGB order.
	"""
	def __call__(self, dataloaderiter, i):
		return (0.0, 0.0, 0.0)
class NvdecodeColorTransform:
	"""
	Interface class that represents a color transformation from YCbCr to RGB as
	defined in Benzina's kernel.

	0: ITU-R BT.601-6-625 recommentation
			Kr = 0.299
			Kg = 0.587
			Kb = 0.114
		but full scale
			Y,Cb,Cr in [0, 255]

	1: ITU-R BT.601-6-625 recommentation
			Kr = 0.299
			Kg = 0.587
			Kb = 0.114
		with head/footroom
			Y       in [16,235]
			Cb,Cr   in [16,240]

	2: ITU-R BT.709 recommentation
			Kr = 0.2126
			Kg = 0.7152
			Kb = 0.0722
		with head/footroom
			Y       in [16,235]
			Cb,Cr   in [16,240]

	3: ITU-R BT.2020 recommentation
			Kr = 0.2627
			Kg = 0.6780
			Kb = 0.0593
		with head/footroom
			Y       in [16,235]
			Cb,Cr   in [16,240]
	"""
	def __init__(self, *args, **kwargs):
		pass
	"""
	__call__ needs to be implemented in subclasses

	Arguments
	---------

	Returns
	-------
	out (tuple):
		a tuple containing a single int indicating which method to use when
		converting from YCbCr to RGB.
	"""
	def __call__(self, dataloaderiter, i):
		return (0,)
class NvdecodeScaleTransform:
	"""
	Interface class that represents a scale transformation.
	"""
	def __init__(self, *args, **kwargs):
		pass
	"""
	__call__ needs to be implemented in subclasses

	Arguments
	---------

	Returns
	-------
	out (tuple):
		a tuple of numerics containing the scale of a sample's RGB channels. It
		should be in RGB order. Components will be multiplied to the respective
		channels of a sample.
	"""
	def __call__(self, dataloaderiter, i):
		return (1.0, 1.0, 1.0)
class NvdecodeBiasTransform:
	"""
	Interface class that represents a bias transformation (rotation, translation,
	scale, skew).
	"""
	def __init__(self, *args, **kwargs):
		pass
	"""
	__call__ needs to be implemented in subclasses

	Arguments
	---------

	Returns
	-------
	out (tuple):
		a tuple containing a single numeric indicating the bias of pixels RGB channels.
		It should be in RGB order. Components will be substracted to the respective
		channels of a sample.
	"""
	def __call__(self, dataloaderiter, i):
		return (0.0, 0.0, 0.0)


class NvdecodeConstantWarpTransform (NvdecodeWarpTransform):
	"""
	Represents a warp transformation to be applied on each sample of a batch
	independently of its index.

	Arguments
	---------
	warp (iterable of numerics, optional): a flatten, row-major 3 x 3 warp matrix
		(default: flatten identity matrix).
	"""
	def __init__(self, warp=None):
		if warp is None:
			warp = (1.0, 0.0, 0.0,
			        0.0, 1.0, 0.0,
			        0.0, 0.0, 1.0)
		self.warp = tuple(warp)
	def __call__(self, dataloaderiter, i):
		return self.warp


class NvdecodeConstantOOBTransform  (NvdecodeOOBTransform):
	"""
	Represents an out of bounds transformation to be applied on each sample of
	a batch independently of its index.

	Arguments
	---------
	oob (numeric or iterable of numerics, optional): an iterable in RGB order
		containing the RGB color to use when no data is available (default:
		``(0.0, 0.0, 0.0)``).
	"""
	def __init__(self, oob=None):
		if   oob is None:
			oob = (0.0, 0.0, 0.0)
		elif isinstance(oob, (int, float)):
			oob = (float(oob),)*3
		self.oob = tuple(oob)
	def __call__(self, dataloaderiter, i):
		return self.oob


class NvdecodeConstantColorTransform(NvdecodeColorTransform):
	"""
	Represents a color transformation to be applied on each sample of a batch
	independently of its index.

	Arguments
	---------
	color (int, optional): the index of the method to use when converting
		from YCbCr to RGB (default: ``0``).
	"""
	def __init__(self, color=None):
		if   color is None:
			color = (0,)
		elif isinstance(color, (int)):
			color = (int(color),)
		self.color = tuple(color)
	def __call__(self, dataloaderiter, i):
		return self.color


class NvdecodeConstantScaleTransform(NvdecodeScaleTransform):
	"""
	Represents a scale transformation to be applied on each sample of a batch
	independently of its index.

	Arguments
	---------
	scale (numeric or iterable of numerics, optional): an iterable in RGB order
		containing the scale of a sample's RGB channels. Components will be multiplied
		to the respective channels of a sample (default: ``(1.0, 1.0, 1.0)``).
	"""
	def __init__(self, scale=None):
		if   scale is None:
			scale = (1.0, 1.0, 1.0)
		elif isinstance(scale, (int, float)):
			scale = (float(scale),)*3
		self.scale = tuple(scale)
	def __call__(self, dataloaderiter, i):
		return self.scale


class NvdecodeConstantBiasTransform (NvdecodeBiasTransform):
	"""
	Represents a bias transformation to be applied on each sample of a batch
	independently of its index.

	Arguments
	---------
	bias (numeric or iterable of numerics, optional): the bias of pixels RGB channels.
	Components will be substracted to the respective channels of a sample
	(default: ``(0.0, 0.0, 0.0)``).
	"""
	def __init__(self, bias=None):
		if   bias is None:
			bias = (0.0, 0.0, 0.0)
		elif isinstance(bias, (int, float)):
			bias = (float(bias),)*3
		self.bias = tuple(bias)
	def __call__(self, dataloaderiter, i):
		return self.bias


class NvdecodeSimilarityTransform   (NvdecodeWarpTransform):
	"""
	Represents a random similarity warp transformation to be applied on each
	sample of a batch.

	Arguments
	---------
	s (iterable of numerics, optional): the scale range to draw a random value from
		(default: ``(+1,+1)``).
	r (iterable of numerics, optional): the rotation range in radian to draw a random
		value from (default: ``(-0,+0)``).
	tx (iterable of numerics, optional): the translation on the x axis range to draw
		a random value from (default: ``(-0,+0)``).
	ty (iterable of numerics, optional): the translation on the y axis range to draw
		a random value from (default: ``(-0,+0)``).
	reflecth (iterable of numerics, optional): the horizontal reflection probability
		range. Valid values are between 0 and 1 (default: ``(0.0)``).
	reflectv (iterable of numerics, optional): the vertical reflection probability
		range. Valid values are between 0 and 1 (default: ``(0.0)``).
	autoscale (bool, optional): If ``True``, the sample will be automatically scaled
		to the output shape before applying the other transformations (default:
		``False``).
	"""
	def __init__(self, s=(+1,+1), r=(-0,+0), tx=(-0,+0), ty=(-0,+0),
	    reflecth=0.0, reflectv=0.0, autoscale=False):
		if isinstance(s, (int, float)):
			s = float(s)
			s = min(s, 1/s)
			s = (s, 1/s)
		
		if isinstance(r, (int, float)):
			r = float(r)
			r = max(r, -r)
			r = (-r, r)
		
		if isinstance(tx, (int, float)):
			tx = float(tx)
			tx = max(tx, -tx)
			tx = (-tx, tx)
		
		if isinstance(ty, (int, float)):
			ty = float(ty)
			ty = max(ty, -ty)
			ty = (-ty, ty)
		
		assert(s[0] > 0 and s[1] > 0)
		
		self.s         = s
		self.r         = r
		self.tx        = tx
		self.ty        = ty
		self.reflecth  = float(reflecth)
		self.reflectv  = float(reflectv)
		self.autoscale = autoscale
	
	def __call__(self, dataloaderiter, i):
		"""Return a random similarity transformation."""
		RNG      = dataloaderiter.RNG
		outshape = dataloaderiter.shape
		inshape  = dataloaderiter.dataset.shape
		s        = np.exp    (RNG.uniform(low = np.log(self.s [0]), high = np.log(self.s [1])))
		r        = np.deg2rad(RNG.uniform(low =        self.r [0],  high =        self.r [1]))
		tx       =            RNG.uniform(low =        self.tx[0],  high =        self.tx[1])
		ty       =            RNG.uniform(low =        self.ty[0],  high =        self.ty[1])
		reflecth = 1-2*(RNG.uniform() < self.reflecth)
		reflectv = 1-2*(RNG.uniform() < self.reflectv)
		
		#
		# H = T_inshape*T*R*S*T_outshape
		#
		T_o_y = (outshape[0]-1)/2
		T_o_x = (outshape[1]-1)/2
		T_outshape = np.asarray([[1, 0, -T_o_x],
		                         [0, 1, -T_o_y],
		                         [0, 0,    1  ]])
		S_y = reflectv/s
		S_x = reflecth/s
		if self.autoscale:
			S_y *= inshape[0]/outshape[0]
			S_x *= inshape[1]/outshape[1]
		S          = np.asarray([[S_x,  0,   0],
		                         [ 0,  S_y,  0],
		                         [ 0,   0,   1]])
		R          = np.asarray([[+np.cos(r), +np.sin(r),   0],
		                         [-np.sin(r), +np.cos(r),   0],
		                         [    0,           0,       1]])
		T_i_y = (inshape[0]-1)/2
		T_i_x = (inshape[1]-1)/2
		T_inshapeT = np.asarray([[1, 0, tx+T_i_x],
		                         [0, 1, ty+T_i_y],
		                         [0, 0,    1  ]])
		
		H = T_inshapeT.dot(R).dot(S).dot(T_outshape)
		
		return tuple(H.flatten().tolist())

