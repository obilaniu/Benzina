
from bitstring import ConstBitStream

from pybenzinaparse import Parser
from pybenzinaparse.utils import find_boxes, find_traks, get_shape, \
    get_sample_table, get_sample_location


class File:
    def __init__(self, disk_file, offset=0):
        self._path = None
        self._disk_file = None
        self._offset = offset
        self._moov = None
        self._moov_pos = None

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
        if not self._moov:
            self._parse()

    def close(self):
        if self._disk_file and not self._disk_file.closed:
            self._disk_file.close()
            self._disk_file = None

    def trak(self, label):
        if self._moov is None:
            raise RuntimeError("File [{}] is missing a moov box"
                               .format(self._disk_file.name if self._disk_file else self._path))

        if isinstance(label, str):
            label = label.encode("utf-8")
        if label[-1] != 0:
            label += b'\0'

        return next(find_traks(self._moov.boxes, label), None)

    def len(self, trak):
        if isinstance(trak, (str, bytes)):
            label = trak
            trak = self.trak(trak)
            if trak is None:
                raise RuntimeError("Could not find trak [{}]".format(label))

        stbl = get_sample_table(trak)
        stsz = next(find_boxes(stbl.boxes, b"stsz"), None)

        return stsz.sample_count

    def shape(self, trak):
        if isinstance(trak, (str, bytes)):
            label = trak
            trak = self.trak(trak)
            if trak is None:
                raise RuntimeError("Could not find trak [{}]".format(label))

        return get_shape(trak)

    def sample_location(self, trak, index):
        if isinstance(trak, (str, bytes)):
            label = trak
            trak = self.trak(label)
            if trak is None:
                raise RuntimeError("Could not find trak [{}]".format(label))

        location = get_sample_location(trak, index)

        if not location:
            return None

        offset, size = location
        return self._offset + offset, size

    def sample_bytes(self, trak, index):
        location = self.sample_location(trak, index)

        if not location:
            return None

        offset, size = location
        self._disk_file.seek(offset)
        return self._disk_file.read(size)

    def sample_as_file(self, trak, index):
        location = self.sample_location(trak, index)

        if not location:
            return None

        offset, _ = location
        return File(self._disk_file, offset=offset)

    def video_configuration_location(self, trak):
        if isinstance(trak, (str, bytes)):
            label = trak
            trak = self.trak(label)
            if trak is None:
                raise RuntimeError("Could not find trak [{}]".format(label))

        stsd = next(find_boxes(get_sample_table(trak).boxes, b"stsd"))
        __c1 = next(find_boxes(stsd.boxes, [b"avc1", b"hec1", b"hvc1"]), None)

        if not __c1:
            return None

        _vcC = next(find_boxes(__c1.boxes, [b"avcC", b"hvcC"]))
        offset, size = (_vcC.header.start_pos + _vcC.header.header_size,
                        _vcC.header.box_size - _vcC.header.header_size)
        return self._moov_pos + offset, size

    def _parse(self):
        self._disk_file.seek(self._offset)
        pos = self._offset

        headers = (Parser.parse_header(ConstBitStream(chunk))
                   for chunk in iter(lambda: self._disk_file.read(32), b''))

        for header in headers:
            if header.type != b"moov":
                pos += header.box_size
                self._disk_file.seek(pos)
                continue

            # Parse MOOV
            self._disk_file.seek(pos)
            moov_bstr = ConstBitStream(self._disk_file.read(header.box_size))
            self._moov = next(Parser.parse(moov_bstr))
            self._moov_pos = pos

            for trak in find_boxes(self._moov.boxes, b"trak"):
                stbl = get_sample_table(trak)
                next(find_boxes(stbl.boxes, [b"stco", b"co64"])).load(moov_bstr)
                next(find_boxes(stbl.boxes, b"stsz")).load(moov_bstr)
            break


class Track:
    def __init__(self, file, label):
        self._file_path = None
        self._file = None
        self._trak = None
        self._len = None
        self._label = label

        # If file is a path, ownership of the file will be held by this instance
        if isinstance(file, str):
            self._file_path = file
        else:
            self._file = file
            self._parse()

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.sample_location(index)

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
        if not self._trak:
            self._parse()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    @property
    def label(self):
        return self._label

    def shape(self):
        return self._file.shape(self._trak)

    def sample_location(self, index):
        return self._file.sample_location(self._trak, index)

    def sample_bytes(self, index):
        return self._file.sample_bytes(self._trak, index)

    def sample_as_file(self, index):
        return self._file.sample_as_file(self._trak, index)

    def video_configuration_location(self):
        return self._file.video_configuration_location(self._trak)

    def _parse(self):
        self._trak = self._file.trak(self._label)
        self._len = self._file.len(self._trak)
