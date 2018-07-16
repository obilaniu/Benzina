# -*- coding: utf-8 -*-
import torch

from torch.utils.data import (Dataset, DataLoader)
from benzina.native   import (BenzinaDatasetCore,
                              BenzinaPluginNvdecodeCore)


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
	- Responsible for creating an BenzinaLoaderIter at every epoch.

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
	def __init__(self,
	             dataset,
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


class BenzinaLoaderIter(object):
	def __init__(self, loader):
		self.dataset        = loader.dataset
		self.multibuffering = loader.multibuffering
		self.shape          = loader.shape
		self.timeout        = loader.timeout
		if   loader.device_id is None or loader.device_id == "cuda":
			self.device_id = torch.device(torch.cuda.current_device())
		elif isinstance(loader.device_id, (str, int)):
			self.device_id = torch.device(loader.device_id)
		else:
			self.device_id = loader.device_id
		self.batch_iter     = iter(loader.batch_sampler)
		self.length         = len(loader)
		self.multibuffer    = None
		self.core           = None
		self.batch_size     = None
	
	def __iter__(self):
		return self
	
	def __len__(self):
		return self.length
	
	def __next__(self):
		if self.core_needs_init():
			self.pull_first_batch()
			self.init_core()
			self.push_first_batch()
			self.fill_core()
		else:
			self.fill_one_batch()
		return self.pull()
	
	def core_needs_init(self):
		return self.core is None
	
	def pull_first_batch(self):
		self.first_batch = next(self.batch_iter)
		
	def init_core(self):
		self.batch_size  = len(self.first_batch)
		self.multibuffer = torch.zeros([self.multibuffering,
		                                self.batch_size,
		                                3,
		                                self.shape[0],
		                                self.shape[1]],
		                               dtype  = torch.float32,
		                               device = self.device_id)
		self.core        = BenzinaPluginNvdecodeCore(
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
		try:
			while self.core.pushes < self.core.multibuffering:
				self.push(next(self.batch_iter))
		except StopIteration: pass
	
	def push(self, batch):
		assert(len(batch) <= self.batch_size)
		buffer = self.multibuffer[self.core.pushes % self.core.multibuffering, :len(batch)]
		
		self.core.defineBatch()
		for s,b in zip(batch, buffer):
			s = int(s)
			# d = self.dataset[s]  # FIXME: Get useful transforms out of this!
			self.core.defineSample     (s, b.data_ptr())
			self.core.setHomography    (1.0, 0.0, 0.0,
			                            0.0, 1.0, 0.0,
			                            0.0, 0.0, 1.0)
			self.core.setBias          (0.0, 0.0, 0.0)
			self.core.setScale         (1.0, 1.0, 1.0)
			self.core.setOOBColor      (0.0, 0.0, 0.0)
			self.core.selectColorMatrix(0)
			self.core.submitSample()
		self.core.submitBatch(buffer)
	
	def pull(self):
		if self.core.pulls >= self.core.pushes:
			raise StopIteration
		return self.core.waitBatch(block=True, timeout=self.timeout)
	
	def fill_one_batch(self):
		try:
			self.push(next(self.batch_iter))
		except StopIteration: pass
	