import numpy as np

from pybenzinaparse.utils import find_headers_at


def find_sample_table_at(file, trak_pos=None):
    # TRAK.MDIA.MINF.STBL
    pos, _, _, header_size = next(find_headers_at(file, {b"trak"}, trak_pos))
    pos, _, _, header_size = next(find_headers_at(file, {b"mdia"}, pos + header_size))
    pos, _, _, header_size = next(find_headers_at(file, {b"minf"}, pos + header_size))
    return next(find_headers_at(file, {b"stbl"}, pos + header_size))


def find_video_configuration_at(file, stbl_pos=None):
    # STBL.STSD.[AVC1 | HEC1 | HVC1].[AVCC | HVCC]
    pos, _, _, header_size = next(find_headers_at(file, {b"stbl"}, stbl_pos))
    pos, _, _, header_size = next(find_headers_at(file, {b"stsd"},
                                                  pos + header_size))
    header_size += (1 +  # version (fullbox): 1 bytes
                    3 +  # flags (fullbox): 24 bits
                    4)   # entry_count: uint32
    __c1 = next(find_headers_at(file,
                                {b"avc1", b"hec1", b"hvc1"},
                                pos + header_size),
                None)

    if __c1 is None:
        return None

    pos, _, _, header_size = __c1

    _vcC_offset = (header_size +
                   6 +      # reserved: 48 bits
                   2 +      # data_reference_index: uint16

                   2 +      # pre_defined: 16 bits
                   2 +      # reserved: 16 bits
                   12 +     # pre_defined: 96 bits

                   2 +      # width: uint16
                   2 +      # height: uint16
                   4 +      # horizresolution: 2 * uint16
                   4 +      # vertresolution: 2 * uint16

                   4 +      # reserved: 32 bits

                   2 +      # frame_count: uint16
                   32 +     # compressorname: 32 bytes
                   2 +      # depth: uint16

                   2)       # pre_defined: 16 bits

    return next(find_headers_at(file, {b"avcC", b"hvcC"}, pos + _vcC_offset))


def get_name_at(file, trak_pos=None):
    # TRAK.MDIA.HDLR
    pos, _, _, header_size = next(find_headers_at(file, {b"trak"}, trak_pos))
    pos, _, _, header_size = next(find_headers_at(file, {b"mdia"}, pos + header_size))
    pos, box_size, _, header_size = next(find_headers_at(file, {b"hdlr"}, pos + header_size))
    name_offset = (header_size +
                   1 +  # version (fullbox): 1 bytes
                   3 +  # flag (fullbox)s: 24 bits
                   4 +  # pre_defined: uint32
                   4 +  # handler_type: 4 bytes
                   12)  # reserved0: 3 * 32 bits
    file.seek(pos + name_offset)
    return file.read(box_size - name_offset)


def get_shape_at(file, trak_pos=None):
    # TRAK.TKHD
    pos, _, _, header_size = next(find_headers_at(file, {b"trak"}, trak_pos))
    pos, box_size, _, header_size = next(find_headers_at(file, {b"tkhd"}, pos + header_size))
    file.seek(pos)
    box_buf = file.read(box_size)
    box_version = box_buf[header_size + 0]  # version (fullbox): uint8
    shape_offset = (header_size +
                    1 +     # version (fullbox): uint8
                    3 +     # flags (fullbox): 24 bits
                    8 +     # creation_time: uint64
                    8 +     # modification_time: uint64
                    4 +     # track_id: uint32
                    4 +     # reserved0: 32 bits
                    8 +     # duration: uint64

                    8 +     # reserved1: 2 * 32 bits

                    2 +     # layer: uint16
                    2 +     # alternate_group: uint16
                    2 +     # volume: 2 * uint8

                    2 +     # reserved2: 16 bits

                    36)     # matrix: 9 * uint32

    if box_version != 1:
        shape_offset += ((4 - 8) +  # creation_time: uint32
                         (4 - 8) +  # modification_time: uint32
                         (4 - 4) +  # track_id: uint32
                         (4 - 4) +  # reserved0: 32 bi32
                         (4 - 8))   # duration: uint32

    width_offset = shape_offset
    height_offset = shape_offset + 4    # width: 2 * uint16
    # Read only integer parts of width and height
    # (width and height are uint16.uint16 floats)
    width = int.from_bytes(box_buf[width_offset:width_offset + 2], "big")
    height = int.from_bytes(box_buf[height_offset:height_offset + 2], "big")
    return width, height


def get_chunk_offset_at(file, stbl_pos=None):
    # STBL.[STCO | CO64]
    pos, _, _, header_size = next(find_headers_at(file, {b"stbl"}, stbl_pos))
    pos, box_size, box_type, header_size = \
        next(find_headers_at(file, {b"stco", b"co64"}, pos + header_size))
    header_size += (1 +     # version (fullbox): 1 bytes
                    3)      # flags (fullbox): 24 bits

    file.seek(pos + header_size +
              4)    # entry_count: uint32
    co_buf = file.read(box_size -
                       header_size -
                       4)   # entry_count: uint32
    co = np.frombuffer(co_buf,
                       np.dtype(">u4") if box_type == b"stco"
                       else np.dtype(">u8"))
    return co, co_buf


def get_sample_size_at(file, stbl_pos=None):
    # STBL.STSZ
    pos, _, _, header_size = next(find_headers_at(file, {b"stbl"}, stbl_pos))
    pos, box_size, box_type, header_size = \
        next(find_headers_at(file, {b"stsz"}, pos + header_size))
    header_size += (1 +     # version (fullbox): 1 bytes
                    3)      # flags (fullbox): 24 bits

    file.seek(pos)
    stsz_buf = file.read(box_size)
    sample_size = int.from_bytes(stsz_buf[header_size:header_size + 4], "big")
    if sample_size > 0:
        sz_buf = stsz_buf[header_size:header_size + 4]
    else:
        sz_buf = stsz_buf[header_size +
                          4 +   # sample_size: uint32
                          4:]   # sample_count: uint32
    sz = np.frombuffer(sz_buf, np.dtype(">u4"))

    return sz, sz_buf
