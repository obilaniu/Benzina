# -*- coding: utf-8 -*-
from collections import namedtuple

import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    _Item = namedtuple("Item", ["input"])

    def __init__(self, track):
        self._track = track

    def __len__(self):
        return len(self._track)
    
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
        return Dataset._Item(self._track.sample_as_file(index))

    def __add__(self, other):
        raise NotImplementedError()


class ImageNet(Dataset):
    _Item = namedtuple("Item", ["input", "target"])

    def __init__(self, input_track, target_track):
        Dataset.__init__(self, input_track)
        self._targets = [None for _ in range(len(input_track))]
        for i in range(len(target_track)):
            self._targets[i] = int.from_bytes(target_track.sample_bytes(i),
                                              byteorder="little")

    def __getitem__(self, index):
        return ImageNet._Item(*Dataset.__getitem__(self, index),
                              self._targets[index])

    def __add__(self, other):
        raise NotImplementedError()

    @property
    def targets(self):
        return self._targets
