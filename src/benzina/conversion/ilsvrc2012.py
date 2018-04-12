# -*- coding: utf-8 -*-

#
# Imports
#
import argparse                  as Ap
import cv2                       as cv
import numpy                     as np
import os
import pdb
import struct
import subprocess
import sys
import time
import uuid
import warnings
warnings.filterwarnings("ignore")
import h5py                      as H
from   psnrhma import psnrhma_color

bp = pdb.set_trace


#
# Utilities
#

def centerImage(img):
	"""
	Select center-square crop of non-square images.
	"""
	
	h,w,c = img.shape
	
	if h>w:
		# Image taller than wide: Chop top and bottom off.
		k = (h-w)//2
		return img[k:k+w,:]
	else:
		# Image wider  than tall: Chop left and right off.
		k = (w-h)//2
		return img[:,k:k+h]

def scaleImage(img, a):
	"""
	Scale image to target size.
	"""
	
	return cv.resize(img,
	                 dsize         = (a.targetscale, a.targetscale),
	                 interpolation = cv.INTER_LANCZOS4)

def compressImage(img, a):
	"""
	Compress the image to a target PSNR or size.
	
	To do this we bisect on x264's Constant Rate Factor, its master quality
	setting. It ranges from 0-51, with 0 being lossless (probably incompatible)
	and 51 being absolute worst. Every -6 decrement roughly corresponds to a
	doubling of the bitrate. 17-19 is considered decent.
	"""
	
	
	if a.targetcrf:
		return doCompressAtCRF(img, a, a.targetcrf)[0]
	
	
	crfDict   = {}
	crfMin    =  1
	crfMax    = 51
	crfCurr   = ((crfMax-crfMin)   // 2) + crfMin
	crfBest   = crfCurr
	crfJump   = max(crfMax-crfCurr, crfCurr-crfMin)
	deltaBest = np.nan
	while True:
		imgH264, imgComp = doCompressAtCRF(img, a, crfCurr)
		crfDict[crfCurr] = {
			"size":    len(imgH264),
			"imgH264": imgH264,
			"imgComp": imgComp,
		}
		
		
		if a.targetsize:
			delta = crfDict[crfCurr]["size"   ] - a.targetsize
		else:
			crfDict[crfCurr]["psnrhma"] = psnrhma_color(img, imgComp)
			delta = crfDict[crfCurr]["psnrhma"] - a.targetpsnrhma
		deltaAbs = abs(delta)
		
		if not deltaBest<deltaAbs:
			deltaBest = deltaAbs
			crfBest   = crfCurr
		
		crfOld  = crfCurr
		crfJump = (crfJump+1) // 2
		if   delta == 0: break                                  # Perfect  quality
		elif delta  < 0: crfCurr = max(crfMin, crfOld-crfJump)  # Increase quality
		else:            crfCurr = min(crfMax, crfOld+crfJump)  # Decrease quality
		if crfJump <= 1 and crfCurr in crfDict: break
		
		if a.verbose:
			print("CRF {} -> {}", crfOld, crfCurr)
			blackbar   = np.zeros((img.shape[0], 10, img.shape[2]), img.dtype)
			sidebyside = np.concatenate([img, blackbar, imgComp], axis=1)
			cv.imshow("Compression", sidebyside)
			cv.waitKey()
	
	if a.verbose:
		print(*["CRF {:2d}  PSNRHMA: {}".format(k, v["psnrhma"]) for k,v in crfDict.items()], sep="\n")
		print("Selecting CRF {}".format(crfBest))
	
	return crfDict[crfBest]["imgH264"]

def doCompressAtCRF(img, a, crf=17):
	"""
	Attempt a compression of the given image at the given CRF.
	
	Return a tuple (compressed image bytes, decompressed image)
	"""
	
	#
	# For lack of programmatic Python access to FFmpeg, resort to the ugly
	# hack of invoking command-lines.
	#
	
	#
	# H264 ENCODER SETTINGS
	# 
	# Ultimately, we don't follow what's below, but it's a collection of notes.
	# There's a lot of intricate detail. In particular, refer to:
	# 
	# - Rec. ITU-R BT.470-6    (11/1998)
	# - Rec. ITU-R BT.601-7    (03/2011)
	# - Rec. ITU-T T.871       (05/2011)
	# - Rec. ITU-T H.264       (10/2016)
	# 
	# 
	# The situation is as follows.
	# 
	#   - We're loading data from JPEG images and transcoding to h264 IDR frames.
	# 
	#   - JFIF (the interchange format for JPEG) requires YCbCr coding of image
	#     data and references the color matrix of "625-line BT601-6", with
	#     modifications that make use of full-range (256) quantization levels.
	# 
	#   - The most popular chroma subsampling method is YUV 4:2:0, meaning that
	#     the U & V chroma samples are subsampled 2x in both horizontal and
	#     vertical directions. In ImageNet there are also YUV 4:4:4-coded images.
	# 
	#   - JFIF's chroma samples are, if subsampled, offset as follows w.r.t. the
	#     luma samples:
	#         Hoff = H_downsample_factor/2 - 0.5
	#         Voff = V_downsample_factor/2 - 0.5
	# 
	#   - Nvidia's NVDEC is only capable of decoding h264 High Profile 4.1
	#     YUV420P in NV12 format, with unknown support for full-scale YUV.
	# 
	#   - FFmpeg will produce I-frame-only video if ctx->gop_size == 0.
	# 
	#   - FFmpeg won't make an encoder context unless it's been told a timebase,
	#     width and height.
	# 
	#   - FFmpeg will force x264 to mark an I-frame as an IDR-frame
	#     (Instantaneous Decoder Refresh) if the option forced_idr == 1.
	# 
	#   - x264 won't shut up unless its log-level is set to none (log=-1)
	# 
	#   - In H264 the above can be coded if:
	#         video_full_range_flag               = 1 (pc/full range)
	#         colour_description_present_flag     = 1
	#         matrix_coefficients                 = 5 (Rec. ITU-R BT.601-6 625)
	#         chroma_format_idc                   = 1 (YUV 4:2:0)
	#         chroma_loc_info_present_flag        = 1
	#         chroma_sample_loc_type_top_field    = 1 (center sample)
	#         chroma_sample_loc_type_bottom_field = 1 (center sample)
	#     Given that the colorspace is that of Rec. ITU-R BT.601-6 625 (PAL), a
	#     reasonable guess is that the transfer characteristics and primaries are
	#     also of that standard, even though they are unspecified in ImageNet:
	#         colour_primaries                    = 5 (Rec. ITU-R BT.601-6 625)
	#         transfer_characteristics            = 1 (Rec. ITU-R BT.601-6 625 is
	#                                                  labelled "6", but "1", which
	#                                                  corresponds to BT.709-5, is
	#                                                  functionally equivalent and
	#                                                  explicitly preferred by the
	#                                                  H264 standard)
	#
	
	try:
		baseName     = str(uuid.uuid4())
		pngPath      = os.path.join(a.tmpDir, baseName+".png")
		compPngPath  = os.path.join(a.tmpDir, baseName+".comp.png")
		h264Path     = os.path.join(a.tmpDir, baseName+".h264")
		cmdConv2H264 = [
			a.ffmpeg,         "-y",
			"-i",             pngPath,
			"-preset",        "placebo",
			"-tune",          "stillimage",
			"-x264-params",   "log=-1:chromaloc=1",
			"-forced-idr",    "1",
			"-g",             "1",
			"-pix_fmt",       "yuv420p",
			"-crf",           str(crf),
			h264Path,
		]
		cmdConv2PNG = [
			a.ffmpeg,         "-y",
			"-i",             h264Path,
			compPngPath,
		]
		
		cv.imwrite(pngPath, img)
		subprocess.check_call(cmdConv2H264,
		                      stdin =subprocess.DEVNULL)
		subprocess.check_call(cmdConv2PNG,
		                      stdin =subprocess.DEVNULL)
		imgComp = cv.imread(compPngPath, cv.IMREAD_COLOR)
		with open(h264Path, "rb") as f:
			imgH264 = f.read()
	finally:
		if os.path.isfile(pngPath):      os.unlink(pngPath)
		if os.path.isfile(h264Path):     os.unlink(h264Path)
		if os.path.isfile(compPngPath):  os.unlink(compPngPath)
	
	return imgH264, imgComp



#
# Main
#
def main(argv):
	argp = Ap.ArgumentParser()
	argp.add_argument("-s", "--srcFile",        default="ilsvrc2012.hdf5",  type=str,
	    help="Path to the HDF5 source file for ImageNet 2012.")
	argp.add_argument("-d", "--dstDir",         default=".",                type=str,
	    help="Path to output datasets directory.")
	argp.add_argument("-t", "--tmpDir",         default=".",                type=str,
	    help="Path to temporary directory.")
	argp.add_argument("--slice",                default=None,               type=int,
	    nargs=2, metavar="N",
	    help="Slice of dataset to handle.")
	argp.add_argument("--targetscale",          default=256,                type=int,
	    help="Target image scale in pixels per side length of square")
	argp.add_argument("--targetcrf",            default=None,               type=int,
	    help="Target Constant Rate Factor (CRF) for x264")
	argp.add_argument("--targetsize",           default=None,               type=int,
	    help="Maximum size of a compressed H264 image, in bytes.")
	argp.add_argument("--targetpsnrhma",        default=40.0,               type=float,
	    help="Target PSNR-HMA corruption, in dB.")
	argp.add_argument("--ffmpeg",               default="/usr/bin/ffmpeg",  type=str,
	    help="FFmpeg binary to invoke.")
	argp.add_argument("-v", "--verbose",        action="store_true",        dest="verbose",
	    help="Whether to display images as the dataset is converted, or not.")
	argp.add_argument("--no-verbose",           action="store_false",       dest="verbose",
		help="Whether not to display images as the dataset is converted.")
	a = argp.parse_args(argv[1:])
	
	assert a.targetsize  is None or a.targetsize  >   0
	assert a.targetscale is None or a.targetscale >= 16
	assert a.targetcrf   is None or (a.targetcrf>=1 and a.targetcrf<=51)
	try:     os.mkdir(a.tmpDir)
	except:  pass
	try:     os.mkdir(a.dstDir)
	except:  pass
	
	#
	# ilsvrc2012.hdf5 contains the following members:
	#
	#    ['encoded_images']:   JPEG files,          as 1431167-long array of uint8 arrays.
	#    ['filenames']:        Original filenames,  as 1431167-long array of strings.
	#    ['targets']:          Target class,        as 1331167-long array of uint16.
	fHDF5 = H.File(a.srcFile, "r")
	
	numExamples         = len(fHDF5['encoded_images'])
	numLabelledExamples = len(fHDF5['targets'])
	baseName            = os.path.basename(a.srcFile)
	baseNameNoExt       = os.path.splitext(baseName)[0]
	if a.slice:
		assert a.slice[0]  < a.slice[1]
		assert a.slice[0] >= 0
		assert a.slice[1] <= numExamples
		fileNameH264 = "{}.{:010d}-{:010d}.h264".format(baseNameNoExt, *a.slice)
	else:
		fileNameH264 = "{}.h264".format(baseNameNoExt)
	pathNameH264      = os.path.join(a.dstDir, fileNameH264)
	pathNameLengths   = os.path.join(a.dstDir, fileNameH264+".lengths")
	pathNameTargets   = os.path.join(a.dstDir, fileNameH264+".targets")
	pathNameFilenames = os.path.join(a.dstDir, fileNameH264+".filenames")
	
	with open(pathNameH264,      "xb") as fH264,    \
	     open(pathNameLengths,   "xb") as fLengths, \
	     open(pathNameTargets,   "xb") as fTargets, \
	     open(pathNameFilenames, "x" ) as fFilenames:
		datasetSlice = range(*a.slice) if a.slice else range(numExamples)
		for i in datasetSlice:
			imgJPEG     = np.array(fHDF5['encoded_images'][i], dtype=np.uint8)
			imgDecoded  = cv.imdecode  (imgJPEG, cv.IMREAD_COLOR)
			imgCentered = centerImage  (imgDecoded)
			imgScaled   = scaleImage   (imgCentered, a)
			imgH264     = compressImage(imgScaled,   a)
			
			fH264   .write(imgH264)
			fLengths.write(struct.pack("<Q", len(imgH264)))
			if i<numLabelledExamples:
				fTargets.write(struct.pack("<q", int(fHDF5['targets'][i])))
			else:
				fTargets.write(struct.pack("<q", int(-1)))
			fFilenames.write(fHDF5["filenames"][i][0].tostring().decode()+"\n")
			fH264   .flush()
			fLengths.flush()
			fTargets.flush()
			fFilenames.flush()
	
	return 0



if __name__ == "__main__":
	sys.exit(main(sys.argv))
