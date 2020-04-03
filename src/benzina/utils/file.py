from collections import namedtuple

from bitstring import ConstBitStream
import numpy as np

from pybenzinaparse import Parser
from pybenzinaparse.utils import find_headers_at, find_boxes

Trak = namedtuple("Trak", ["label", "shape", "stbl", "stbl_pos"])


class File:
    def __init__(self, disk_file, offset=0):
        self._path = None
        self._disk_file = None
        self._offset = offset
        self._traks = None

        # If disk_file is a path, ownership of the disk file will be held by
        # this instance
        if isinstance(disk_file, str):
            self._path = disk_file
        else:
            self._disk_file = disk_file
            self._parse()

    def __enter__(self):
        # If self._path is set, ownership of the disk file is held by this
        # instance and it will be opened here
        if self._path:
            self.open()
        return self

    def __exit__(self, type, value, traceback):
        # If self._path is set, ownership of the disk file is held by this
        # instance and the file should be closed here
        if self._path:
            self.close()

    @property
    def path(self):
        return self._path

    @property
    def offset(self):
        return self._offset

    def open(self):
        if not self._disk_file or self._disk_file.closed:
            self._disk_file = open(self._path, "rb")
        if self._traks is None:
            self._parse()

    def close(self):
        if self._disk_file and not self._disk_file.closed:
            self._disk_file.close()
            self._disk_file = None

    def seek(self, offset):
        self._disk_file.seek(offset)

    def read(self, size):
        return self._disk_file.read(size)

    def trak(self, label):
        if self._traks is None:
            raise RuntimeError("File [{}] is missing a moov box"
                               .format(self._disk_file.name if self._disk_file else self._path))

        if isinstance(label, str):
            label = label.encode("utf-8")

        if label[-1] == 0:
            label = label[:-1]

        return self._traks.get(label, None)

    def subfile(self, offset):
        return File(self._disk_file, offset=offset)

    def _parse(self):
        self._traks = {}

        for moov_pos, moov in find_headers_at(self._disk_file, {b"moov"}, self._offset):
            for trak_pos, trak in find_headers_at(self._disk_file, {b"trak"},
                                                  moov_pos + moov.header_size, moov.box_size - moov.header_size):
                tkhd_pos, tkhd = next(find_headers_at(self._disk_file, {b"tkhd"}, trak_pos + trak.header_size))
                self._disk_file.seek(tkhd_pos)
                tkhd_bstr = ConstBitStream(self._disk_file.read(tkhd.box_size))
                tkhd = next(Parser.parse(tkhd_bstr))
                trak_shape = tkhd.width, tkhd.height

                mdia_pos, mdia = next(find_headers_at(self._disk_file, {b"mdia"}, trak_pos + trak.header_size))

                hdlr_pos, hdlr = next(find_headers_at(self._disk_file, {b"hdlr"}, mdia_pos + mdia.header_size))
                self._disk_file.seek(hdlr_pos)
                hdlr_bstr = ConstBitStream(self._disk_file.read(hdlr.box_size))
                trak_label = next(Parser.parse(hdlr_bstr)).name

                if trak_label[-1] == 0:
                    trak_label = trak_label[:-1]

                minf_pos, minf = next(find_headers_at(self._disk_file, {b"minf"}, mdia_pos + mdia.header_size))
                stbl_pos, stbl = next(find_headers_at(self._disk_file, {b"stbl"}, minf_pos + minf.header_size))
                self._disk_file.seek(stbl_pos)
                stbl_bstr = ConstBitStream(self._disk_file.read(stbl.box_size))
                trak_stbl = next(Parser.parse(stbl_bstr))
                trak_stbl_pos = stbl_pos

                self._traks[trak_label] = Trak(trak_label, trak_shape, trak_stbl, trak_stbl_pos)
            break


class Track:
    def __init__(self, file, label):
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
        if isinstance(file, str):
            self._file_path = file
        else:
            self._file = file
            self._parse()

    def __len__(self):
        return self._len

    def __getitem__(self, index):
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
        if not self._file:
            self._file = File(self._file_path)
        self._file.open()
        if self._trak is None:
            self._parse()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

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
        if index < 0:
            index = len(self) + index
        return int(self._file.offset + self._co[index]), int(self._sz[index])

    def sample_bytes(self, index):
        offset, size = self.sample_location(index)
        self._file.seek(offset)
        return self._file.read(size)

    def sample_as_file(self, index):
        offset, _ = self.sample_location(index)
        return self._file.subfile(offset)

    def video_configuration_location(self):
        stsd = next(find_boxes(self._trak.stbl.boxes, {b"stsd"}))
        c1 = next(find_boxes(stsd.boxes, {b"avc1", b"hec1", b"hvc1"}), None)

        if not c1:
            return None

        cC = next(find_boxes(c1.boxes, {b"avcC", b"hvcC"}))
        return (self._trak.stbl_pos + cC.header.start_pos + cC.header.header_size,
                cC.header.box_size - cC.header.header_size)

    def _parse(self):
        self._trak = self._file.trak(self._label)

        stbl, stbl_pos = self._trak.stbl, self._trak.stbl_pos

        co_box = next(find_boxes(stbl.boxes, {b"stco", b"co64"}))
        self._file.seek(stbl_pos +
                        co_box.header.start_pos + co_box.header.header_size +
                        4)  # entry_count
        self._co_buffer = self._file.read(co_box.header.box_size -
                                          co_box.header.header_size -
                                          4)    # entry_count

        self._co = np.frombuffer(self._co_buffer,
                                 np.dtype(">u4") if co_box.header.type == b"stco"
                                 else np.dtype(">u8"))

        sz_box = next(find_boxes(stbl.boxes, {b"stsz"}))
        if sz_box.sample_size > 0:
            self._sz = np.full(co_box.entry_count, sz_box.sample_size, np.uint32)
        else:
            self._file.seek(stbl_pos +
                            sz_box.header.start_pos + sz_box.header.header_size +
                            4 +  # sample_size
                            4)   # sample_count
            self._sz_buffer = self._file.read(sz_box.header.box_size -
                                              sz_box.header.header_size -
                                              4 -  # sample_size
                                              4)   # sample_count
            self._sz = np.frombuffer(self._sz_buffer, np.dtype(">u4"))

        self._len = co_box.entry_count


class Sample:
    def __init__(self, track, index):
        self._track = track
        self._index = index
        pass

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
