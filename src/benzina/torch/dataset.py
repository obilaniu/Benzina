# -*- coding: utf-8 -*-
from collections import namedtuple
import os
import typing

import numpy as np
import torch.utils.data

from benzina.utils.file import Track


_TrackType = typing.Union[str, Track]
_TrackPairType = typing.Tuple[_TrackType, _TrackType]
_ClassTracksType = typing.Tuple[_TrackType, _TrackType]


class Dataset(torch.utils.data.Dataset):
    """
    Args:
        archive (str or :class:`Track`): path to the archive or a Track. If a
            Track, :attr:`track` will be ignored.
        track (str or :class:`Track`, optional): track label or a Track. If a
            Track, :attr:`archive` must not be specified.
            (default: ``"bzna_input"``)
    """

    _Item = namedtuple("Item", ["input"])

    def __init__(self,
                 archive: typing.Union[str, _TrackType] = None,
                 track: _TrackType = "bzna_input"):
        if isinstance(archive, Track):
            track = archive
            archive = None

        if archive is not None:
            if not isinstance(track, str):
                raise ValueError("track option must be a track label when "
                                 "archive is specified.")

            archive = os.path.expanduser(archive)
            archive = os.path.expandvars(archive)

            if not os.path.isfile(archive):
                raise ValueError("The archive {} is not present.".format(archive))

            track = Track(archive, track)

        elif not isinstance(track, Track):
            raise ValueError("track option must be a Track when archive is "
                             "not specified.")

        self._track = track
        if self._track.file.closed:
            # Parse metadata then re-close the file handle
            self._track.open()
            self._track.close()

        self._filename = track.file.name

    @property
    def filename(self):
        return self._filename

    def __len__(self):
        return len(self._track)
    
    def __getitem__(self, index: int):
        return Dataset._Item(self._track[index])

    def __add__(self, other):
        raise NotImplementedError()


class ClassificationDataset(Dataset):
    """
    Args:
        archive (str or pair of :class:`Track`): path to the archive or a pair
            of Track. If a pair of Track, :attr:`tracks` will be ignored.
        tracks (pair of str or :class:`Track`, optional): pair of input and
            target tracks labels or a pair of input and target Track. If a pair
            of Track, :attr:`archive` must not be specified.
            (default: ``("bzna_input", "bzna_target")``)
        input_label (str, optional): label of the inputs to use in the input
            track. (default: ``"bzna_thumb"``)
    """

    _Item = namedtuple("Item", ["input", "input_label", "aux"])

    def __init__(self,
                 archive: typing.Union[str, _TrackPairType] = None,
                 tracks: _ClassTracksType = ("bzna_input", "bzna_target"),
                 input_label: str = "bzna_thumb"):
        try:
            archive, tracks = \
                ClassificationDataset._validate_source(None, archive)
        except (TypeError, ValueError):
            archive, tracks = \
                ClassificationDataset._validate_source(archive, tracks)

        if archive is not None:
            input_track = Track(archive, tracks[0])
            target_track = Track(archive, tracks[1])

        else:
            input_track, target_track = tracks

        Dataset.__init__(self, input_track)

        self._indices = np.arange(Dataset.__len__(self), dtype=np.int64)

        is_target_track_closed = target_track.file.closed
        if is_target_track_closed:
            target_track.open()
        location_first, _ = target_track[0].location
        location_last, size_last = target_track[-1].location
        target_track.file.seek(location_first)
        buffer = target_track.file.read(location_last + size_last - location_first)
        if is_target_track_closed:
            target_track.close()

        self._targets = np.full(len(self._track), -1, np.int64)
        self._targets[:len(target_track)] = np.frombuffer(buffer, np.dtype("<i8"))

        self._input_label = input_label

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index: int):
        item = Dataset.__getitem__(self, self._indices[index])
        return ClassificationDataset._Item(input=item.input,
                                           input_label=self._input_label,
                                           aux=self._targets[index])

    def __add__(self, other):
        raise NotImplementedError()

    @property
    def targets(self):
        return self._targets

    @staticmethod
    def _validate_source(*args):
        archive, tracks = args

        if archive is not None:
            if any(not isinstance(t, str) for t in tracks):
                raise ValueError("tracks option must be a pair of tracks "
                                 "labels when archive is specified.")

            archive = os.path.expanduser(archive)
            archive = os.path.expandvars(archive)

            if not os.path.isfile(archive):
                raise ValueError("The archive {} is not present.".format(archive))

            _, _ = tracks

        elif any(not isinstance(t, Track) for t in tracks):
            raise ValueError("tracks option must be a pair of Track when "
                             "archive is not specified.")

        return archive, tracks


class ImageNet(ClassificationDataset):
    """
    Args:
        root (str or pair of :class:`Track`): root of the ImageNet dataset or
            path to the archive or a pair of Track. If a pair of Track,
            :attr:`tracks` will be ignored.
        split (None or str, optional): The dataset split, supports ``test``,
            ``train``, ``val``. If not specified, samples will be drawn from
            all splits.
        tracks (pair of str or :class:`Track`, optional): pair of input and
            target tracks labels or a pair of input and target Track. If a pair
            of Track, :attr:`root` must not be specified.
            (default: ``("bzna_input", "bzna_target")``)
        input_label (str, optional): label of the inputs to use in the input
            track. (default: ``"bzna_thumb"``)
    """

    # Some images are missing from the dataset. Please read the README of the
    # dataset for more information.
    LEN_VALID = 50000 - 1
    LEN_TEST = 100000 - 7

    def __init__(self,
                 root: typing.Union[str, _TrackPairType] = None,
                 split: str = None,
                 tracks: _ClassTracksType = ("bzna_input", "bzna_target"),
                 input_label: str = "bzna_thumb"):
        try:
            archive, split, tracks = \
                ImageNet._validate_source(None, split, root)
        except (TypeError, ValueError):
            archive, split, tracks = \
                ImageNet._validate_source(root, split, tracks)

        ClassificationDataset.__init__(self, archive, tracks, input_label)

        if split == "test":
            self._indices = self._indices[-self.LEN_TEST:]
            self._targets = self._targets[-self.LEN_TEST:]

        elif split == "train":
            len_train = len(self) - self.LEN_VALID - self.LEN_TEST
            self._indices = self._indices[:len_train]
            self._targets = self._targets[:len_train]

        elif split == "val":
            len_train = len(self) - self.LEN_VALID - self.LEN_TEST
            self._indices = self._indices[len_train:-self.LEN_TEST]
            self._targets = self._targets[len_train:-self.LEN_TEST]

    def __add__(self, other):
        raise NotImplementedError()

    @staticmethod
    def _validate_source(*args):
        root, split, tracks = args

        archive = None

        if root is not None:
            if any(not isinstance(t, str) for t in tracks):
                raise ValueError("tracks option must be a pair of tracks "
                                 "labels when root is specified.")

            root = os.path.expanduser(root)
            root = os.path.expandvars(root)

            if os.path.isfile(root):
                archive = root

            elif os.path.isfile(os.path.join(root, "ilsvrc2012.bzna")):
                archive = os.path.join(root, "ilsvrc2012.bzna")

            elif os.path.isfile(os.path.join(root, "ilsvrc2012.mp4")):
                archive = os.path.join(root, "ilsvrc2012.mp4")

            if archive is None:
                if root.endswith(".mp4") or root.endswith(".bzna"):
                    raise ValueError("The archive {} is not present.".format(root))

                else:
                    raise ValueError("The archive ilsvrc2012.[mp4|bzna] is not "
                                     "present in root {}.".format(root))

        elif any(not isinstance(t, Track) for t in tracks):
            raise ValueError("tracks option must be a pair of Track when "
                             "root is not specified.")

        if split not in {"test", "train", "val", None}:
            raise ValueError("split option must be one of test, train, val")

        return archive, split, tracks
