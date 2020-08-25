from datetime import datetime
import io

from pybenzinaparse import headers, utils

from benzina.utils import mp4


def test_find_headers_at():
    creation_time = utils.to_mp4_time(datetime(2019, 9, 15, 0, 0, 0))
    modification_time = utils.to_mp4_time(datetime(2019, 9, 16, 0, 0, 0))

    samples_sizes = [198297, 127477, 192476]
    samples_offset = 10
    trak = utils.make_trak(creation_time, modification_time,
                           samples_sizes, samples_offset)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]
    tkhd.track_id = 1
    tkhd.width = [512, 0]
    tkhd.height = [512, 0]

    trak.refresh_box_size()

    buffer = io.BytesIO(bytes(trak))
    for (pos, box_size, box_type, header_size), \
        box in zip(mp4.find_headers_at(buffer, {b'tkhd', b'mdia'},
                                       offset=trak.header.header_size),
                   utils.find_boxes(trak.boxes, {b'tkhd', b'mdia'})):
        assert pos < buffer.tell()
        assert box_size == box.header.box_size
        assert box_type == box.header.type
        if isinstance(box.header, headers.FullBoxHeader):
            assert header_size + 4 == box.header.header_size
        else:
            assert header_size == box.header.header_size


def test_find_sample_table_at():
    creation_time = utils.to_mp4_time(datetime(2019, 9, 15, 0, 0, 0))
    modification_time = utils.to_mp4_time(datetime(2019, 9, 16, 0, 0, 0))

    samples_sizes = [198297, 127477, 192476]
    samples_offset = 10
    trak = utils.make_trak(creation_time, modification_time,
                           samples_sizes, samples_offset)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]
    tkhd.track_id = 1
    tkhd.width = [512, 0]
    tkhd.height = [512, 0]

    trak.refresh_box_size()

    buffer = io.BytesIO(bytes(trak))
    _, box_size, box_type, header_size = mp4.find_sample_table_at(buffer, 0)
    stbl = utils.get_sample_table(trak)
    assert box_size == stbl.header.box_size
    assert box_type == stbl.header.type
    assert header_size == stbl.header.header_size


def test_get_chunk_offset_at():
    creation_time = utils.to_mp4_time(datetime(2019, 9, 15, 0, 0, 0))
    modification_time = utils.to_mp4_time(datetime(2019, 9, 16, 0, 0, 0))

    samples_sizes = [198297, 127477, 192476]
    samples_offset = 10
    trak = utils.make_trak(creation_time, modification_time,
                           samples_sizes, samples_offset)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]
    tkhd.track_id = 1
    tkhd.width = [512, 0]
    tkhd.height = [512, 0]

    trak.refresh_box_size()

    buffer = io.BytesIO(bytes(trak))
    stbl_pos, _, _, _ = mp4.find_sample_table_at(buffer, 0)
    co, _ = mp4.get_chunk_offset_at(buffer, stbl_pos)
    stbl = utils.get_sample_table(trak)
    co_box = next(utils.find_boxes(stbl.boxes, {b"stco", b"co64"}))
    assert (co == [e.chunk_offset for e in co_box.entries]).all()


def test_get_sample_size_at():
    creation_time = utils.to_mp4_time(datetime(2019, 9, 15, 0, 0, 0))
    modification_time = utils.to_mp4_time(datetime(2019, 9, 16, 0, 0, 0))

    samples_sizes = [198297, 127477, 192476]
    samples_offset = 10
    trak = utils.make_trak(creation_time, modification_time,
                           samples_sizes, samples_offset)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]
    tkhd.track_id = 1
    tkhd.width = [512, 0]
    tkhd.height = [512, 0]

    trak.refresh_box_size()

    buffer = io.BytesIO(bytes(trak))
    stbl_pos, _, _, _ = mp4.find_sample_table_at(buffer, 0)
    sz, _ = mp4.get_sample_size_at(buffer, stbl_pos)
    stbl = utils.get_sample_table(trak)
    stsz = next(utils.find_boxes(stbl.boxes, {b"stsz"}))
    assert (sz == [e.entry_size for e in stsz.samples]).all()


def test_get_name_at():
    creation_time = utils.to_mp4_time(datetime(2019, 9, 15, 0, 0, 0))
    modification_time = utils.to_mp4_time(datetime(2019, 9, 16, 0, 0, 0))

    samples_sizes = [198297, 127477, 192476]
    samples_offset = 10
    trak = utils.make_trak(creation_time, modification_time,
                           samples_sizes, samples_offset)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]
    tkhd.track_id = 1
    tkhd.width = [512, 0]
    tkhd.height = [512, 0]

    trak.refresh_box_size()

    buffer = io.BytesIO(bytes(trak))
    assert mp4.get_name_at(buffer, 0) == utils.get_name(trak)


def test_get_shape_at():
    creation_time = utils.to_mp4_time(datetime(2019, 9, 15, 0, 0, 0))
    modification_time = utils.to_mp4_time(datetime(2019, 9, 16, 0, 0, 0))

    samples_sizes = [198297, 127477, 192476]
    samples_offset = 10
    trak = utils.make_trak(creation_time, modification_time,
                           samples_sizes, samples_offset)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]
    tkhd.track_id = 1
    tkhd.width = [512, 0]
    tkhd.height = [512, 0]

    trak.refresh_box_size()

    buffer = io.BytesIO(bytes(trak))
    assert mp4.get_shape_at(buffer, 0) == utils.get_shape(trak)
