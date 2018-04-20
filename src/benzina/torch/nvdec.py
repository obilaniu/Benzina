# -*- coding: utf-8 -*-
import torch
import torchvision

from torch.utils.data import (Dataset, DataLoader)
from benzina.native   import (BenzinaDatasetCore,
                              BenzinaLoaderIterCore)


"""

C library
	- Responsible for loading and initializing libnvcuvid.so.

C/Python extension:
	- Responsible for facading the C library.

BenzinaDataset:
	- Responsible for mmap()'ing and cudaHostRegister()'ing or not the huge
	  memory slab.
	- Responsible for reporting useful data from __getitem__().

BenzinaLoader:
	- Responsible for binding together:
		- Dataset
		- Sampler
		- The ID of one GPU device. (In a distant future, a CPU backend might
		  exist too.)
	- Responsible for creating an NVDECLoaderIter at every epoch.

BenzinaLoaderIter:
	- Responsible for allocating suitable N-buffer
	- Responsible for allocating one CUvideodecoder per dataset.
	- Responsible for spawning 1 worker thread per CUvideodecoder, ready to
	  receive commands into ring buffer protected by pthreads semaphore or
	  condvar+mutex and pipelining the decode and colorspace conversion.
	- Responsible for spawning 1 IO thread that receives read commands into
	  ring buffer.
	- Responsible for spawning a thread that asynchronously enqueues work onto
	  the IO thread.
	- Responsible for yielding one of the parts of the N-buffer,
	  blocking until it is filled.
"""


class BenzinaDataset(Dataset)
	def __init__(self, root):
		self._core = BenzinaDatasetCore(root)
	
	def __len__(self):
		return self._core.length
	
	def __getitem__(self, index):
		#
		# It's actually BenzinaLoaderIter's responsibility to translate indices
		# into asynchronously-loaded images.
		#
		return (self, index)
	
	@property
	def shape(self):
		"""Shape of images in dataset, as (h,w) tuple."""
		return (self._core.h, self._core.w)


class BenzinaLoader(DataLoader):
	def __init__(dataset,
	             batch_size     = 1,
	             shuffle        = False,
	             sampler        = None,
	             batch_sampler  = None,
	             drop_last      = False,
	             timeout        = 0,
	             multibuffering = 3,
	             device_id      = None,
	             shape          = None):
		super().__init__(dataset,
		                 batch_size     = batch_size,
		                 shuffle        = shuffle,
		                 sampler        = sampler,
		                 batch_sampler  = batch_sampler,
		                 num_workers    = 0,
		                 collate_fn     = None,
		                 pin_memory     = True,
		                 drop_last      = drop_last,
		                 timeout        = float(timeout),
		                 worker_init_fn = None)
		
		if device_id is None:
			device_id = torch.cuda.current_device()
		
		self.device_id      = device_id
		self.shape          = shape or dataset.shape
		self.multibuffering = multibuffering
	
	def __iter__(self):
		return BenzinaLoaderIter(self)


class BenzinaLoaderIter(object):
	def __init__(self, loader):
		self.loader         = loader
		self.dataset        = self.loader.dataset
		self.multibuffering = self.loader.multibuffering
		self.shape          = self.loader.shape
		self.device_id      = self.loader.device_id
		self.sample_iter    = iter(self.loader.batch_sampler)
		
		self.multibuffer    = None
		self._core          = None
		self.batch_size     = None # Will only become known during first __next__() call.
		
		self.batch_exps     = len(loader)
		self.batch_reqs     = 0
		self.batch_rets     = 0
		self.batch_exc      = None
	
	def __del__(self):
		if self._core:
			self._core.shutdown()
			self._core = None
		self.multibuffer = None
	
	def __len__(self):
		return len(self.loader)
	
	def __iter__(self):
		return self
	
	def __next__(self):
		"""Get next batch."""
		
		# BATCH REQUESTS. OPERATES IN A MANNER DECOUPLED FROM BATCH RETURNS SIDE.
		if self.batch_reqs == 0:
			#
			# First call is special. We detect the batch size, initialize core
			# and fill the pipeline.
			#
			indices    = next(self.sample_iter)
			batch_size = len(indices)
			self._init(batch_size)
			self._enqueue_batch(indices)
			try:
				while self.batch_reqs < self.multibuffering:
					self._enqueue_batch(next(self.sample_iter))
			except StopIteration as self.batch_exc:
				pass
		else:
			try:
				self._enqueue_batch(next(self.sample_iter))
			except StopIteration as self.batch_exc:
				pass
		
		# BATCH RETURNS. OPERATES IN A MANNER DECOUPLED FROM BATCH REQUESTS SIDE.
		if self.batch_rets >= self.batch_exps:
			raise self.batch_exc
		else:
			return self._dequeue_batch()
	
	def _init(self, batch_size):
		"""Initialize C Core."""
		self.batch_size  = batch_size
		self.multibuffer = torch.ones([self.multibuffering,
		                               self.batch_size,
		                               3,
		                               self.shape[0],
		                               self.shape[1]],
		                              dtype  = torch.float32,
		                              device = torch.device(self.device_id))
		self._core = BenzinaLoaderIterCore(self)
	
	def _enqueue_batch(self, indices):
		"""Enqueue a batch's worth of indices to work on."""
		self._core.begin_batch()
		for i in indices:
			self._core.enqueue_sample(self.dataset._core,
			                          #
			                          # Homography Perspective Distortion Matrix
			                          1.000000, 0.000000, 0.000000,
			                          0.000000, 1.000000, 0.000000,
			                          0.000000, 0.000000, 1.000000,
			                          #
			                          # YUV->RGB Colorspace Conversion Matrix
			                          1.000000, 0.000000, 0.000000,
			                          0.000000, 1.000000, 0.000000,
			                          0.000000, 0.000000, 1.000000)
		self._core.end_batch()
		self.batch_reqs += 1
	
	def _dequeue_batch(self):
		"""Block until next batch becomes ready."""
		subbuffer = self.batch_rets % self.multibuffering
		self._core.wait()
		batch = ()
		self.batch_rets += 1
		return batch


