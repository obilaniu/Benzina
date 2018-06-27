# -*- coding: utf-8 -*-
import threading
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


class BenzinaDataset(Dataset):
	def __init__(self, root):
		self._core = BenzinaDatasetCore(root)
	
	def __len__(self):
		return len(self._core)
	
	def __getitem__(self, index):
		#
		# This does not return images; Rather, it returns a tuple
		# (index, byteOffset, byteLength). It's actually BenzinaLoaderIter's
		# responsibility to translate indices into asynchronously-loaded
		# images.
		#
		return self._core[index]
	
	@property
	def shape(self):
		"""Shape of images in dataset, as (h,w) tuple."""
		return self._core.shape


class BenzinaLoader(DataLoader):
	def __init__(dataset,
	             batch_size     = 1,
	             shuffle        = False,
	             sampler        = None,
	             batch_sampler  = None,
	             drop_last      = False,
	             timeout        = 0,
	             shape          = None,
	             device_id      = None,
	             multibuffering = 3):
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
		
		self.device_id      = device_id or 0
		self.shape          = shape     or dataset.shape
		self.multibuffering = multibuffering
	
	def __iter__(self):
		return BenzinaLoaderIter(self)


class SynchronousPipelineIter(object):
	def __init__(self, stages=3):
		self._stages      = stages
		self._insnFetched = 0
		self._insnRetired = 0
	
	def __iter__(self):
		return self
	
	def __next__(self):
		try:                  self.pushFrontend(self.pullFrontEnd())
		except StopIteration: self.doomFrontend()
		return                self.pushBackEnd (self.pullBackEnd ())
	
	@property
	def doomed(self):
		return hasattr(self, "_insnDoomed")
	
	@property
	def dead(self):
		return self.doomed and self._insnRetired >= self._insnDoomed
	
	def pullFrontend(self):
		pass
	
	def pushFrontend(self, *args, **kwargs):
		#
		# When subclassing, it's critical to do the super() call *AFTER* the
		# actual work is done.
		#
		if self.doomed: return
		self._insnFetched += 1
		while self._insnFetched<self._stages:
			self.pushFrontend(self.pullFrontEnd())
	
	def pullBackend (self):
		if self.dead:
			raise StopIteration
		else:
			self._insnRetired += 1
	
	def pushBackend (self, *args, **kwargs):
		pass
	
	def doomFrontend(self):
		if not self.doomed:
			self._insnDoomed = self._insnFetched


class BenzinaLoaderIter(SynchronousPipelineIter):
	def __init__(self, loader):
		self.dataset        = loader.dataset
		self.multibuffering = loader.multibuffering
		self.shape          = loader.shape
		self.deviceId       = loader.device_id
		if self.deviceId is None:
			self.deviceId = torch.cuda.current_device()
		self.sampleIter     = enumerate(loader.batch_sampler)
		self.length         = len(loader)
		self.multibuffer    = None
		self._core          = None
		# Will only become known after first next(self.samplerIter) call.
		self.batchSize      = None
		
		super().__init__(self.multibuffering)
	
	def __len__(self):
		return self.length
	
	def pullFrontend(self):
		if self.doomed: raise  StopIteration
		else:           return next(self.sample_iter)
	
	def pushFrontend(self, i_indices):
		if self.doomed: return
		if not self._core:
			self._init(i_indices[1])
		self._core.push(**self.getJob(i_indices))
		super().pushFrontend()
	
	def pullBackend (self):
		if self.dead: raise StopIteration
		batch = self._core.pull()
		super().pullBackend()
		return batch
	
	def pushBackend (self, batch):
		return batch
	
	def doomFrontend(self):
		if self.doomed: return
		super().doomFrontend()
		self.sampleIter = None
	
	def getJob      (self, i_indices):
		i, indices = i_indices
		
		batchSize  = len(indices)
		buffer     = self.multibuffer[i % self.multibuffering][:batchSize]
		
		intIndices = []
		offsets    = []
		lengths    = []
		cudaPtrs   = []
		for idx in indices:
			offset, length = self.dataset[idx]
			cudaPtr        = buffer[idx].data_ptr()
			intIndices.append(int(idx))
			offsets   .append(offset)
			lengths   .append(length)
			cudaPtrs  .append(cudaPtr)
		
		return {
			"batchSize":  batchSize,
			"indices":    np.array(intIndices, dtype=np.uint64).tobytes(),
			"offsets":    np.array(offsets,    dtype=np.uint64).tobytes(),
			"lengths":    np.array(lengths,    dtype=np.uint64).tobytes(),
			"cudaPtrs":   np.array(cudaPtrs,   dtype=np.uint64).tobytes(),
			"H":          np.eye(3, dtype=np.float32)                    \
			                [np.newaxis, ...]                            \
			                .repeat(batchSize, axis=0)                   \
			                .copy("C")                                   \
			                .tobytes(),
			"C":          np.eye(3, dtype=np.float32)                    \
			                [np.newaxis, ...]                            \
			                .repeat(batchSize, axis=0)                   \
			                .copy("C")                                   \
			                .tobytes(),
			"B":          np.zeros((batchSize, 3), dtype=np.float32)     \
			                .copy("C")                                   \
			                .tobytes(),
			"aux":        (buffer,),
		}
	
	def shutdown    (self):
		if not self.doomed:
			self.doomFrontend()
		
		if self._core:
			self._core.shutdown()
			self._core = None
			self.multibuffer = None
	
	def _init       (self, indices):
		"""Initialize C Core."""
		self.batchSize   = len(indices)
		self.multibuffer = torch.ones([self.multibuffering,
		                               self.batchSize,
		                               3,
		                               self.shape[0],
		                               self.shape[1]],
		                              dtype  = torch.float32,
		                              device = torch.device(self.deviceId))
		
		#
		# The core needs to know:
		#   - BenzinaDatasetCore
		#   - Device
		#   - Multibuffering depth
		#   - Maximum batch size
		#   - Target image shape
		#
		self._core = BenzinaLoaderIterCore(self.dataset._core,
		                                   self.deviceId,
		                                   self.multibuffering,
		                                   self.batchSize,
		                                   self.shape[0],
		                                   self.shape[1])
