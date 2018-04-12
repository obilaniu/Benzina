# -*- coding: utf-8 -*-

"""
Internal Design


Dataset Files:

  - NAME.h264
    
        Primary image data file, h264-coded. Resident on disk except if
        preloaded whole on an exceptionally-large memory machine.
        
        Format: Concatenated h264 images (IDR frames) constituting a video.
        
        The file is in principle playable as a video as-is.
    
  - NAME.h264.lengths
    
        Flat array of the length in bytes of each h264 image file in
        NAME.h264, as a uint64. There are as many uint64's in the file as
        there are images Resident in memory. For fast seeking purposes;
        A cumsum() in memory can make this table even more useful.
        Resident in memory.
    
  - NAME.h264.targets
    
        Flat array of the target class of the corresponding h264 image,
        as an int64 from 0 to #CLASSES - 1. If the class is not known
        (test set, or unsupervised), the target is -1. Resident in memory.
    
  - NAME.h264.filenames
      
        The originating filename from which the corresponding h264 image was
        transcoded, as a string, one per line. Probably not useful in general.
    
  - NAME.h264.nvdecode
    
        Precomputed NVDECODE data for loading. Flat array with same number of
        NVDECODE argument structs as there are images in NAME.h264.
        Resident in memory. Pointers into the payload are precomputed as
        NULL+offset from beginning of each h264 image. For relocation,
        increment pointers by the value of the base pointer for that h264 image.
  
  - NAME.meta.pkl
    
        Pickle file containing metadata about the dataset, for use by Python
        code. Resident in memory.



Tools:
	1) Converter to h264 (Python)
	2) NVDECODE precalculation (C)
	3) Server (C)
"""


import zmq

