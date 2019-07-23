# -*- coding: utf-8 -*-
import benzina.native
import gc
import numpy                           as np
import os, sys
import torch
import torch.utils.data

from   torch.utils.data.dataloader import default_collate
from   contextlib                  import suppress
from   .                           import operations as ops



class DataLoader(torch.utils.data.DataLoader):
    """
    Loads images from a benzina.torch.dataset.Dataset. Encapsulates a sampler
    and data processing transformations.

    Args:
        dataset (benzina.torch.dataset.Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled at every
            epoch (default: False).
        sampler (torch.utils.data.Sampler, optional): defines the strategy to draw
            samples from the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (torch.utils.data.Sampler, optional): like sampler, but returns
            a batch of indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a
            batch. Should always be non-negative. (default: 0)
        shape (int or tuple of ints, optional): set the shape of the samples. Note
            that this does not imply a resize of the image but merely set the shape
            of the tensor in which the data will be copied.
        device (torch.device, optional): set the device to use. Note that only CUDA
            devices are supported for the moment. (default: None)
        multibuffering (int, optional): set the size of the multibuffering buffer.
            (default: 3).
        seed (int, optional): set the seed for the random transformations.
        warp_transform (benzina.torch.operations.WarpTransform or iterable of float, optional):
            set the warp transformation or use as the arguments to initialize a
            WarpTransform.
        norm_transform (benzina.torch.operations.NormTransform or float or iterable of float, optional):
            set the normalization transformation. Values to multiply a pixel's channels
            with. Note that this transformation is applied after bias_transform.
        bias_transform (benzina.torch.operations.BiasTransform or float, optional):
            set the bias transformation. Values to substract a pixel's channels with.
            Note that this transformation is applied after norm_transform.
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
                 device          = None,
                 multibuffering  = 3,
                 seed            = None,
                 warp_transform  = None,
                 norm_transform  = None,
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
        
        if not isinstance(warp_transform, ops.WarpTransform):
            warp_transform = ops.ConstantWarpTransform(warp_transform)
        if not isinstance(norm_transform, ops.NormTransform):
            norm_transform = ops.ConstantNormTransform(norm_transform)
        if not isinstance(bias_transform, ops.BiasTransform):
            bias_transform = ops.ConstantBiasTransform(bias_transform)
        
        self.device          = device
        self.multibuffering  = multibuffering
        self.shape           = shape
        self.RNG             = np.random.RandomState(seed)
        self.warp_transform  = warp_transform
        self.color_transform = ops.ConstantColorTransform()
        self.oob_transform   = ops.ConstantOOBTransform()
        self.norm_transform  = norm_transform
        self.bias_transform  = bias_transform
    
    def __iter__(self):
        return _DataLoaderIter(self)


class _DataLoaderIter:
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
        if   loader.device is None or loader.device == "cuda":
            self.device = torch.device(torch.cuda.current_device())
        elif isinstance(loader.device, (str, int)):
            self.device = torch.device(loader.device)
        else:
            self.device = loader.device
        self.RNG             = np.random.RandomState(loader.RNG.randint(2**32))
        self.warp_transform  = loader.warp_transform
        self.color_transform = loader.color_transform
        self.oob_transform   = loader.oob_transform
        self.norm_transform  = loader.norm_transform
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
                                       device = self.device)
        self.core        = benzina.native.NvdecodeDataLoaderIterCore(
                               self.dataset._core,
                               str(self.device),
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
        t_args  = (self.dataset.shape, self.shape, self.RNG)
        
        with self.core.batch(token) as batch:
            for i,ptr in zip(indices, ptrs):
                with batch.sample(i, ptr):
                    self.core.setHomography    (*self.warp_transform (i, *t_args))
                    self.core.selectColorMatrix(*self.color_transform(i, *t_args))
                    self.core.setBias          (*self.bias_transform (i, *t_args))
                    self.core.setScale         (*self.norm_transform (i, *t_args))
                    self.core.setOOBColor      (*self.oob_transform  (i, *t_args))
    
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
