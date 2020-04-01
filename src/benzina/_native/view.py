from . import _C



class FileView(_C.FileView):
    #
    # Required __slots__:
    #    fd:      uint64 (Unix: int fd; Windows: void* handle)
    #    fdoff:   uint64 (bytes)
    #    membase: void*
    #    memtail: void*
    #    parent:  PyObject*
    #
    
    def __new__(kls, *args, **kwargs):
        self = super().__new__(kls, *args, **kwargs)
        return self
    
    def __init__(self, *args, **kwargs):
        pass
    

class BoxViewftyp(_C.BoxView):
    pass
class BoxViewmdat(_C.BoxView):
    pass
class BoxViewmoov(_C.BoxView):
    pass
class BoxViewmeta(_C.BoxView):
    pass
class BoxViewmvhd(_C.BoxView):
    pass
class BoxViewtrak(_C.BoxView):
    pass
class BoxViewtkhd(_C.BoxView):
    pass
class BoxViewtref(_C.BoxView):
    pass
class BoxViewtrgr(_C.BoxView):
    pass
class BoxViewedts(_C.BoxView):
    pass
class BoxViewmdia(_C.BoxView):
    pass
class BoxViewmdhd(_C.BoxView):
    pass
class BoxViewhdlr(_C.BoxView):
    pass
class BoxViewminf(_C.BoxView):
    pass
class BoxViewvmhd(_C.BoxView):
    pass
class BoxViewsmhd(_C.BoxView):
    pass
class BoxViewhmhd(_C.BoxView):
    pass
class BoxViewsthd(_C.BoxView):
    pass
class BoxViewnmhd(_C.BoxView):
    pass
class BoxViewdinf(_C.BoxView):
    pass
class BoxViewdref(_C.BoxView):
    pass
class BoxViewurl(_C.BoxView):
    pass
class BoxViewurn(_C.BoxView):
    pass
class BoxViewstbl(_C.BoxView):
    pass
class BoxViewstsd(_C.FullBoxView):
    pass
class BoxViewstts(_C.FullBoxView):
    pass
class BoxViewstsc(_C.FullBoxView):
    pass
class BoxViewstco(_C.FullBoxView):
    pass
class BoxViewco64(_C.FullBoxView):
    pass
class BoxViewstsz(_C.BoxView):
    pass
class BoxViewstss(_C.FullBoxView):
    pass
class BoxViewstsh(_C.FullBoxView):
    pass
class BoxViewctts(_C.FullBoxView):
    pass
class BoxViewcslg(_C.FullBoxView):
    pass
class BoxViewavcC(_C.BoxView):
    pass
class BoxViewhvcC(_C.BoxView):
    pass


