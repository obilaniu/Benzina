# -*- coding: utf-8 -*-
import benzina.native
import numpy            as np
import os
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
	def __init__(self, root):
		self._root = os.path.abspath(root)
		self._check_have_dir()
		self._check_have_file("data.bin")
		self._check_have_file("data.lengths")
		self._check_have_file("data.nvdecode")
		self._check_have_file("data.protobuf")
		self._check_have_file("README.md")
		self._check_have_file("SHA256SUMS")
		self._core = benzina.native.DatasetCore(self.root)
	
	def __len__(self):
		return len(self._core)
	
	def __getitem__(self, index):
		#
		# This does not return images; Rather, it returns a tuple of some kind,
		# e.g. (index, byteOffset, byteLength). The iterator will *not* use
		# this method for image loading, since it can directly access the
		# dataset core and translate indices into asynchronously-loaded images.
		#
		# This should be overriden in a subclass to return e.g. labels or
		# target information.
		#
		return self._core[index]
	
	@property
	def root(self):
		"""Absolute path to dataset directory."""
		return self._root
	
	@property
	def shape(self):
		"""Shape of images in dataset, as (h,w) tuple."""
		return self._core.shape
	
	def _check_have_dir(self, *paths):
		filePath = os.path.join(self.root, *paths)
		if not os.path.isdir(filePath):
			raise FileNotFoundError(filePath)
	
	def _check_have_file(self, *paths):
		filePath = os.path.join(self.root, *paths)
		if not os.path.isfile(filePath):
			raise FileNotFoundError(filePath)


class ImageNet(Dataset):
	def __init__(self, root):
		super().__init__(root)
		self._check_have_file("data.filenames")
		self._check_have_file("data.targets")
		with open(os.path.join(self.root, "data.targets"), "r") as f:
			self.targets = np.fromfile(f, np.dtype("<i8"))
	
	def __getitem__(self, index):
		return (int(self.targets[index]),)
