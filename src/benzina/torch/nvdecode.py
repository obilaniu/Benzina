# -*- coding: utf-8 -*-
import benzina.native
import numpy                           as np
import os
import torch
import torch.utils.data

from   torch.utils.data.dataloader import default_collate
from   contextlib                  import suppress



class NvdecodeDataLoader(torch.utils.data.DataLoader):
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
		if not isinstance(bias_transform,  NvdecodeWarpTransform):
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
	
	def __iter__(self):
		return self
	
	def __len__(self):
		return self.length
	
	def __next__(self):
		if self.stop_iteration is not None:
			raise self.stop_iteration
		try:
			if self.core_needs_init():
				self.pull_first_batch()
				self.init_core()
				self.push_first_batch()
				with suppress(StopIteration):
					self.fill_core()
			else:
				with suppress(StopIteration):
					self.fill_one_batch()
			return self.pull()
		except StopIteration as si:
			self.core           = None
			self.multibuffer    = None
			self.stop_iteration = si
			raise self.stop_iteration
	
	def core_needs_init(self):
		return self.core is None
	
	def pull_first_batch(self):
		self.first_batch = next(self.batch_iter)
		
	def init_core(self):
		self.check_or_set_batch_size(self.first_batch)
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
	
	def push_first_batch(self):
		self.push(self.__dict__.pop("first_batch"))
	
	def fill_core(self):
		while self.core.pushes < self.core.multibuffering:
			self.fill_one_batch()
	
	def push(self, batch):
		self.check_or_set_batch_size(batch)
		buffer = self.multibuffer[self.core.pushes % self.core.multibuffering, :len(batch)]
		
		aux = []
		self.core.defineBatch()
		for s,b in zip(batch, buffer):
			s = int(s)
			aux.append(self.dataset[s])
			self.core.defineSample     (s, int(b.data_ptr()))
			self.core.setHomography    (*self.warp_transform (self, s))
			self.core.selectColorMatrix(*self.color_transform(self, s))
			self.core.setBias          (*self.bias_transform (self, s))
			self.core.setScale         (*self.scale_transform(self, s))
			self.core.setOOBColor      (*self.oob_transform  (self, s))
			self.core.submitSample()
		aux = self.collate_fn(aux)
		self.core.submitBatch((buffer, *aux))
	
	def pull(self):
		if self.core.pulls >= self.core.pushes:
			raise StopIteration
		return self.core.waitBatch(block=True, timeout=self.timeout)
	
	def fill_one_batch(self):
		self.push(next(self.batch_iter))
	
	def check_or_set_batch_size(self, iter_batch):
		iter_batch_size = len(iter_batch)
		
		if   self.batch_size is None:
			self.batch_size = iter_batch_size
		elif self.batch_size < iter_batch_size:
			raise RuntimeError("Batch size expected to be {}, but iterator returned larger batch size {}!"
			                   .format(self.batch_size, iter_batch_size))
		elif self.batch_size > iter_batch_size:
			if self.drop_last:
				raise StopIteration


class NvdecodeWarpTransform:
	def __init__(self, *args, **kwargs):
		pass
	def __call__(self, dataloaderiter, i):
		return (1.0, 0.0, 0.0,
		        0.0, 1.0, 0.0,
		        0.0, 0.0, 1.0)
class NvdecodeOOBTransform:
	def __init__(self, *args, **kwargs):
		pass
	def __call__(self, dataloaderiter, i):
		return (0.0, 0.0, 0.0)
class NvdecodeColorTransform:
	def __init__(self, *args, **kwargs):
		pass
	def __call__(self, dataloaderiter, i):
		return (0,)
class NvdecodeScaleTransform:
	def __init__(self, *args, **kwargs):
		pass
	def __call__(self, dataloaderiter, i):
		return (1.0, 1.0, 1.0)
class NvdecodeBiasTransform:
	def __init__(self, *args, **kwargs):
		pass
	def __call__(self, dataloaderiter, i):
		return (0.0, 0.0, 0.0)


class NvdecodeConstantWarpTransform (NvdecodeWarpTransform):
	def __init__(self, warp=None):
		if warp is None:
			warp = (1.0, 0.0, 0.0,
			        0.0, 1.0, 0.0,
			        0.0, 0.0, 1.0)
		self.warp = tuple(warp)
	def __call__(self, dataloaderiter, i):
		return self.warp


class NvdecodeConstantOOBTransform  (NvdecodeOOBTransform):
	def __init__(self, oob=None):
		if   oob is None:
			oob = (0.0, 0.0, 0.0)
		elif isinstance(oob, (int, float)):
			oob = (float(oob),)*3
		self.oob = tuple(oob)
	def __call__(self, dataloaderiter, i):
		return self.oob


class NvdecodeConstantColorTransform(NvdecodeColorTransform):
	def __init__(self, color=None):
		if   color is None:
			color = (0,)
		elif isinstance(color, (int)):
			color = (int(color),)
		self.color = tuple(color)
	def __call__(self, dataloaderiter, i):
		return self.color


class NvdecodeConstantScaleTransform(NvdecodeScaleTransform):
	def __init__(self, scale=None):
		if   scale is None:
			scale = (1.0, 1.0, 1.0)
		elif isinstance(scale, (int, float)):
			scale = (float(scale),)*3
		self.scale = tuple(scale)
	def __call__(self, dataloaderiter, i):
		return self.scale


class NvdecodeConstantBiasTransform (NvdecodeBiasTransform):
	def __init__(self, bias=None):
		if   bias is None:
			bias = (0.0, 0.0, 0.0)
		elif isinstance(bias, (int, float)):
			bias = (float(bias),)*3
		self.bias = tuple(bias)
	def __call__(self, dataloaderiter, i):
		return self.bias


class NvdecodeSimilarityTransform   (NvdecodeWarpTransform):
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

