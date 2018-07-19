# -*- coding: utf-8 -*-
import benzina.native
import os
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
	def __init__(self, root):
		self._root = os.path.abspath(root)
		self._check_have_dir()
		self._check_have_file("data.bin")
		self._check_have_file("data.lengths")
		self._check_have_file("data.protobuf")
		self._check_have_file("README.md")
		self._check_have_file("SHA256SUMS")
		self._core = benzina.native.DatasetCore(self.root)
	
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
