from collections import namedtuple

import numpy as np

from benzina.utils.mp4 import get_chunk_offset_at, get_name_at, \
    get_sample_size_at, get_shape_at, find_headers_at, \
    find_sample_table_at, find_video_configuration_at

FileDesc = namedtuple("FileDesc", ["name", "mode"])
Trak = namedtuple("Trak", ["label", "shape", "stbl_pos"])


class FileReadMixin:
    def __init__(self, disk_file=None, *_args, **_kwargs):
        self._file = disk_file

    @property
    def closed(self):
        return self._file is None or self._file.closed

    def seek(self, offset):
        self._file.seek(offset)

    def read(self, size):
        return self._file.read(size)


class FileProxy(FileReadMixin):
    """ When closed, this proxy can be serialized without problems.
    """
    def __init__(self, disk_file, mode="rb"):
        super().__init__(None, mode=mode)
        self._name = None
        self._mode = mode

        if isinstance(disk_file, str):
            self._name = disk_file
        else:
            try:
                self._name = disk_file.name
                self._file = disk_file
            except AttributeError:
                self._name = None
                self._file = disk_file

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def name(self):
        return self._name

    @property
    def mode(self):
        return self._mode

    def open(self):
        if self.closed:
            self._file = open(self._name, mode=self._mode)

    def close(self):
        if not self.closed and self._name is not None:
            self._file.close()
            self._file = None


class File(FileReadMixin):
    def __init__(self, disk_file, offset=0):
        super().__init__(disk_file, offset=offset)
        self._owner = False
        self._offset = offset
        self._traks = None

        # If disk_file is a path, ownership of the disk file will be held by
        # this instance
        if isinstance(disk_file, str):
            self._owner = True
        if not isinstance(self._file, FileProxy):
            self._file = FileProxy(disk_file, mode="rb")
        if not self.closed:
            self._parse()

    def __enter__(self):
        # If self._path is set, ownership of the disk file is held by this
        # instance and it will be opened here
        if self._owner:
            self.open()
        return self

    def __exit__(self, type, value, traceback):
        # If self._path is set, ownership of the disk file is held by this
        # instance and the file should be closed here
        if self._owner:
            self.close()

    @property
    def name(self):
        return self._file.name

    @property
    def offset(self):
        return self._offset

    def open(self):
        self._file.open()
        if self._traks is None:
            self._parse()

    def close(self):
        self._file.close()

    def trak(self, label):
        if self._traks is None:
            raise RuntimeError("File [{}] is missing a moov box"
                               .format(self._file.name))

        if isinstance(label, str):
            label = label.encode("utf-8")

        if label[-1] == 0:
            label = label[:-1]

        return self._traks.get(label, None)

    def subfile(self, offset):
        return File(self._file, offset=offset)

    def _parse(self):
        self._traks = {}

        moov_pos, box_size, _, header_size = \
            next(find_headers_at(self._file, {b"moov"}, self._offset))
        self._file.seek(moov_pos)
        for trak_pos, _, _, _ in \
            find_headers_at(self._file, {b"trak"},
                            moov_pos + header_size, box_size - header_size):
            trak_label = get_name_at(self._file, trak_pos)

            if trak_label[-1] == 0:
                trak_label = trak_label[:-1]

            trak_shape = get_shape_at(self._file, trak_pos)
            trak_stbl_pos, _, _, _ = find_sample_table_at(self._file, trak_pos)

            self._traks[trak_label] = Trak(trak_label,
                                           trak_shape,
                                           trak_stbl_pos)


class Track:
    def __init__(self, f, label):
        self._file_path = None
        self._file = None
        self._label = label
        self._trak = None

        self._len = None
        self._co_buffer = None
        self._co = np.empty(0, np.uint64)
        self._sz_buffer = None
        self._sz = np.empty(0, np.uint32)

        # If file is a path, ownership of the file will be held by this instance
        if isinstance(f, str):
            self._file_path = f
            self._file = File(self._file_path)
        else:
            self._file = f
            if not self._file.closed:
                self._parse()

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [Sample(self, i) for i in range(*index.indices(len(self)))]
        elif index >= len(self):
            raise IndexError
        return Sample(self, index)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __enter__(self):
        # If self._file_path is set, ownership of the file is held by this
        # instance and it will be opened here
        if self._file_path:
            self.open()
        return self

    def __exit__(self, type, value, traceback):
        # If self._file_path is set, ownership of the file is held by this
        # instance and it should be closed here
        if self._file_path:
            self.close()

    def open(self):
        self._file.open()
        if self._trak is None:
            self._parse()

    def close(self):
        self._file.close()

    @property
    def file(self):
        return self._file

    @property
    def label(self):
        return self._label

    @property
    def shape(self):
        return self._trak.shape

    def sample_location(self, index):
        return int(self._file.offset + self._co[index]), int(self._sz[index])

    def sample_bytes(self, index):
        self.open()
        offset, size = self.sample_location(index)
        self._file.seek(offset)
        return self._file.read(size)

    def sample_as_file(self, index):
        offset, _ = self.sample_location(index)
        return self._file.subfile(offset)

    def video_configuration_location(self):
        _vcC = find_video_configuration_at(self._file, self._trak.stbl_pos)

        if _vcC is None:
            return None

        pos, box_size, _, header_size = _vcC

        return pos + header_size, box_size - header_size

    def _parse(self):
        trak = self._file.trak(self._label)

        self._co, self._co_buffer = \
            get_chunk_offset_at(self._file, trak.stbl_pos)
        self._sz, self._sz_buffer = \
            get_sample_size_at(self._file, trak.stbl_pos)

        self._co, self._sz =  np.broadcast_arrays(self._co, self._sz)

        self._len = len(self._co)
        self._trak = trak


class Sample:
    def __init__(self, track, index):
        self._track = track
        self._index = index

    def __bytes__(self):
        return self._track.sample_bytes(self._index)

    @property
    def value(self):
        return bytes(self)

    @property
    def location(self):
        return self._track.sample_location(self._index)

    def as_file(self):
        return self._track.sample_as_file(self._index)
