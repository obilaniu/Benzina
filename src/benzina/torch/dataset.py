# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np
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
        return Dataset._Item(self._track[index])

    def __add__(self, other):
        raise NotImplementedError()


class ImageNet(Dataset):
    _Item = namedtuple("Item", ["input", "target"])

    def __init__(self, input_track, target_track):
        Dataset.__init__(self, input_track)

        location_first, _ = target_track[0].location
        location_last, size_last = target_track[-1].location
        target_track.file.seek(location_first)
        buffer = target_track.file.read(location_last + size_last - location_first)

        self._targets = np.full(len(input_track), -1, np.int64)
        self._targets[:len(target_track)] = np.frombuffer(buffer, np.dtype("<i8"))

    def __getitem__(self, index):
        return ImageNet._Item(*Dataset.__getitem__(self, index),
                              (self.targets[index],))

    def __add__(self, other):
        raise NotImplementedError()

    @property
    def targets(self):
        return self._targets
