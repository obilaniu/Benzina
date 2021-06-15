import struct
from collections import namedtuple

# ELF Class (32/64-bit)
ELFCLASS32 = 1
ELFCLASS64 = 2

# ELF endianness (LE/BE)
ELFDATA2LSB = 1
ELFDATA2MSB = 2

# ELF version
#   (currently, only one exists: v1)
EV_CURRENT = 1

# ELF object type
#   (Usually relocatable object, executable, or dynamic shared object;
#    more uncommonly, a core dump)
ET_NONE = 0
ET_REL  = 1
ET_EXEC = 2
ET_DYN  = 3
ET_CORE = 4

# ELF Program Header Type (segment type)
PT_NULL    = 0
PT_LOAD    = 1
PT_DYNAMIC = 2
PT_INTERP  = 3
PT_NOTE    = 4
PT_SHLIB   = 5
PT_PHDR    = 6
PT_TLS     = 7

# ELF DYNAMIC segment tag entry
DT_NULL       = 0
DT_NEEDED     = 1
DT_PLTRELSZ   = 2
DT_PLTGOT     = 3
DT_HASH       = 4
DT_STRTAB     = 5
DT_SYMTAB     = 6
DT_RELA       = 7
DT_RELASZ     = 8
DT_RELAENT    = 9
DT_STRSZ      = 10
DT_SYMENT     = 11
DT_INIT       = 12
DT_FINI       = 13
DT_SONAME     = 14
DT_RPATH      = 15
DT_SYMBOLIC   = 16
DT_REL        = 17
DT_RELSZ      = 18
DT_RELENT     = 19
DT_PLTREL     = 20
DT_DEBUG      = 21
DT_TEXTREL    = 22
DT_JMPREL     = 23
DT_ENCODING   = 32
OLD_DT_LOOS   = 0x60000000
DT_LOOS       = 0x6000000d
DT_HIOS       = 0x6ffff000
DT_VALRNGLO   = 0x6ffffd00
DT_VALRNGHI   = 0x6ffffdff
DT_ADDRRNGLO  = 0x6ffffe00
DT_ADDRRNGHI  = 0x6ffffeff
DT_VERSYM     = 0x6ffffff0
DT_RELACOUNT  = 0x6ffffff9
DT_RELCOUNT   = 0x6ffffffa
DT_FLAGS_1    = 0x6ffffffb
DT_VERDEF     = 0x6ffffffc
DT_VERDEFNUM  = 0x6ffffffd
DT_VERNEED    = 0x6ffffffe
DT_VERNEEDNUM = 0x6fffffff
OLD_DT_HIOS   = 0x6fffffff
DT_LOPROC     = 0x70000000
DT_HIPROC     = 0x7fffffff

# ELF DT_FLAG_1 flags
DF_1_NOW        = 0x00000001
DF_1_GLOBAL     = 0x00000002
DF_1_GROUP      = 0x00000004
DF_1_NODELETE   = 0x00000008
DF_1_LOADFLTR   = 0x00000010
DF_1_INITFIRST  = 0x00000020
DF_1_NOOPEN     = 0x00000040
DF_1_ORIGIN     = 0x00000080
DF_1_DIRECT     = 0x00000100
DF_1_TRANS      = 0x00000200
DF_1_INTERPOSE  = 0x00000400
DF_1_NODEFLIB   = 0x00000800
DF_1_NODUMP     = 0x00001000
DF_1_CONFALT    = 0x00002000
DF_1_ENDFILTEE  = 0x00004000
DF_1_DISPRELDNE = 0x00008000
DF_1_DISPRELPND = 0x00010000
DF_1_NODIRECT   = 0x00020000
DF_1_IGNMULDEF  = 0x00040000
DF_1_NOKSYMS    = 0x00080000
DF_1_NOHDR      = 0x00100000
DF_1_EDITED     = 0x00200000
DF_1_NORELOC    = 0x00400000
DF_1_SYMINTPOSE = 0x00800000
DF_1_GLOBAUDIT  = 0x01000000
DF_1_SINGLETON  = 0x02000000
DF_1_STUB       = 0x04000000
DF_1_PIE        = 0x08000000


class ELF:
    def __init__(self, fp, mode='r+b'):
        self.elf_class_64  = None
        self.little_endian = None
        self.file_handle   = open(fp, mode) if isinstance(fp, str) else fp
        header = self.file_handle.read(8)
        
        assert header[:4] == b'\x7fELF'
        assert header[4] in {ELFCLASS32,  ELFCLASS64}
        assert header[5] in {ELFDATA2LSB, ELFDATA2MSB}
        assert header[6] in {EV_CURRENT}
        
        self.elf_class_64  = header[4] == ELFCLASS64
        self.little_endian = header[5] == ELFDATA2LSB
        self.close         = self.file_handle.close
        
        if self.elf_class_64:
            self.e_ehdr_fmt = 'BBBBBBBBBBBBBBBBHHIQQQIHHHHHH'
            self.e_phdr_fmt = 'IIQQQQQQ'
            self.e_phdr_tup = namedtuple('Elf64_Phdr', [
                'p_type',  'p_flags',  'p_offset', 'p_vaddr',
                'p_paddr', 'p_filesz', 'p_memsz',  'p_align',
            ])
        else:
            self.e_ehdr_fmt = 'BBBBBBBBBBBBBBBBHHIIIIIHHHHHH'
            self.e_phdr_fmt = 'IIIIIIII'
            self.e_phdr_tup = namedtuple('Elf32_Phdr', [
                'p_type',   'p_offset', 'p_vaddr', 'p_paddr',
                'p_filesz', 'p_memsz',  'p_flags', 'p_align',
            ])
        
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, \
        self.e_type,  self.e_machine, self.e_version, \
        self.e_entry, self.e_phoff,   self.e_shoff, \
        self.e_flags, \
        self.e_ehsize, \
        self.e_phentsize, self.e_phnum, \
        self.e_shentsize, self.e_shnum, \
        self.e_shstrndx = self.rdfmt(self.e_ehdr_fmt, off=0)
    
    def __enter__(self):                 return self
    def __exit__(self, *args, **kwargs): self.close()
    
    @property
    def struct_fmt(self):
        return '<' if self.little_endian else '>'
    def pack(self, format, *args, **kwargs):
        return struct.pack(self.struct_fmt+format, *args, **kwargs)
    def unpack(self, format, buffer):
        return struct.unpack(self.struct_fmt+format, buffer)
    
    def seek(self, off=None):
        if off is not None:
            self.file_handle.seek(off)
    
    def rdbytes(self, length=None, off=None):
        self.seek(off)
        return self.file_handle.read(length)
    
    def wrbytes(self, v, off=None):
        self.seek(off)
        if not v:
            return
        self.file_handle.write(v)
    
    def rdfmt(self, format, off=None):
        sz = struct.calcsize(self.struct_fmt+format)
        return self.unpack(format, self.rdbytes(sz,off))
    
    def wrfmt(self, off, format, *args, **kwargs):
        sz = struct.calcsize(self.struct_fmt+format)
        self.wrbytes(self.pack(format, *args, **kwargs), off)
    
    def rd8u (self, off=None): return self.unpack('B', self.rdbytes(1,off))[0]
    def rd8s (self, off=None): return self.unpack('b', self.rdbytes(1,off))[0]
    def rd16u(self, off=None): return self.unpack('H', self.rdbytes(2,off))[0]
    def rd16s(self, off=None): return self.unpack('h', self.rdbytes(2,off))[0]
    def rd32u(self, off=None): return self.unpack('I', self.rdbytes(4,off))[0]
    def rd32s(self, off=None): return self.unpack('i', self.rdbytes(4,off))[0]
    def rd64u(self, off=None): return self.unpack('Q', self.rdbytes(8,off))[0]
    def rd64s(self, off=None): return self.unpack('q', self.rdbytes(8,off))[0]
    
    def wr8u (self, v, off=None): self.wrbytes(self.pack('B', v & (2**8 -1)), off)
    def wr8s (self, v, off=None): self.wr8u   (v, off)
    def wr16u(self, v, off=None): self.wrbytes(self.pack('H', v & (2**16-1)), off)
    def wr16s(self, v, off=None): self.wr16u  (v, off)
    def wr32u(self, v, off=None): self.wrbytes(self.pack('I', v & (2**32-1)), off)
    def wr32s(self, v, off=None): self.wr32u  (v, off)
    def wr64u(self, v, off=None): self.wrbytes(self.pack('Q', v & (2**64-1)), off)
    def wr64s(self, v, off=None): self.wr64u  (v, off)
    
    def rdphdr(self, off=None):
        return self.e_phdr_tup._make(self.rdfmt(self.e_phdr_fmt, off))
    
    def iterphdr(self):
        for off in range(self.e_phoff,
                         self.e_phoff+self.e_phnum*self.e_phentsize,
                         self.e_phentsize):
            yield off, self.rdphdr(off)
