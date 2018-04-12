#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import cv2           as cv
import numpy         as np
import scipy.fftpack


#
# Contrast Sensitivity Function coefficients and masking factors.
#

CSFCof  = np.array(
          [[1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
           [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
           [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
           [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
           [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
           [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
           [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
           [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950]]
          );

MaskCof = np.array(
          [[0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
           [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
           [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
           [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
           [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
           [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
           [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
           [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]]
          );

MaskC0f = MaskCof.copy()
MaskC0f[0,0] = 0


#
# PSNR-HMA
#
# http://ponomarenko.info/psnrhma.htm
#
# N. Ponomarenko, O. Ieremeiev, V. Lukin, K. Egiazarian, M. Carli, Modified
# Image Visual Quality Metrics for Contrast Change and Mean Shift Accounting,
# Proceedings of CADSM, February 2011, Ukraine, pp. 305 - 311.
#
# This is a metric that compares two images in an attempt at measuring the
# visual quality of one image w.r.t. another. It attempts to more closely tail
# human perception than naive metrics like MSE or PSNR.
#

def psnrhma(reference, corrupted, eps=1e-30):
	"""Computes the PSNR-HMA's between an (array of) reference and corrupted
	image(s).
	
	Both inputs are of the same shape, (...,H,W) with H=image height and
	W=image width, whose pixels take on values in the range 0-255. The output
	is a tensor shaped (...) giving the PSNR-HMA in dB for each feature map.
	"""
	
	img1        = reference.astype("float64")
	img2        = corrupted.astype("float64")
	rdxAxes     = (-2,-1)
	T           = np.prod (img1.shape[-2:])
	delt        = np.mean (img1-img2, axis=rdxAxes, keepdims=True)
	img2m       = img2+delt
	mean1       = np.mean (img1,  axis=rdxAxes, keepdims=True)
	mean2       = np.mean (img2m, axis=rdxAxes, keepdims=True)
	tmp         = (img1-mean1)*(img2m-mean2)
	sq          = np.var  (img2m, axis=rdxAxes, keepdims=True)
	l           = np.sum  (tmp,   axis=rdxAxes, keepdims=True)/T/sq
	KofContr    = np.where(l<1, np.full_like(l, 0.002),
	                            np.full_like(l, 0.250))
	img3m       = mean2 + (img2m-mean2)*l
	A           = blockImg(img1)
	B           = blockImg(img2m)
	B2          = blockImg(img3m)
	A_dct       = dct2(A)
	B_dct       = dct2(B)
	B2_dct      = dct2(B2)
	MaskA       = maskeff(A, A_dct)
	MaskB       = maskeff(B, B_dct)
	MaskA       = np.maximum(MaskA, MaskB)
	uTmp        = np.abs(A_dct-B_dct)
	u2Tmp       = np.abs(A_dct-B2_dct)
	u           = np.maximum(uTmp  - MaskA/MaskCof, 0)
	u2          = np.maximum(u2Tmp - MaskA/MaskCof, 0)
	u [...,0,0] = uTmp [...,0,0]
	u2[...,0,0] = u2Tmp[...,0,0]
	S1          = np.sum((u *CSFCof)**2, axis=(-2,-1)).sum(axis=(-2,-1), keepdims=True)
	SS1         = np.sum((u2*CSFCof)**2, axis=(-2,-1)).sum(axis=(-2,-1), keepdims=True)
	S1         /= np.prod(A.shape[-4:])
	SS1        /= np.prod(A.shape[-4:])
	delt       *= delt
	S1          = np.where(S1>SS1, SS1+(S1-SS1)*KofContr, S1)
	S1         += 0.04*delt
	S1          = S1.sum(axis=(-2,-1))
	return        20*np.log10(255) - 10*np.log10(S1+eps)


#
# RGB color version of PSNR-HMA.
#
# Since the human eye is less sensitive to chroma than luma, averaging the
# PSNR-HMA of the red, green and blue planes wouldn't be fair.
#
# Instead we convert from the RGB to the YcbCr colorspace, then give full
# weight to the luma PSNR-HMA and half weight to the chroma PSNR-HMAs.
#

def psnrhma_color(reference, corrupted, eps=1e-30):
	a = cv.cvtColor(reference, cv.COLOR_BGR2YCrCb).transpose(2,0,1)
	b = cv.cvtColor(corrupted, cv.COLOR_BGR2YCrCb).transpose(2,0,1)
	
	u = psnrhma(a, b, eps)
	
	p = 255*255/(10.0**(u/10.0))
	
	s = (p[0] + (p[1]+p[2])*0.5)/2
	
	return 20*np.log10(255) - 10*np.log10(s)

#
# Reshape (an array of) image(s) into a grid of 8x8 blocks.
#

def blockImg(img):
	H, W = img.shape[-2:]
	return np.swapaxes(img.reshape(img.shape[:-2]+(H//8, 8, W//8, 8)), -3, -2)

#
# 8x8 2D DCT-II with orthogonal scaling on last two axes.
#

def dct2(blockedImg):
	blockedImg = scipy.fftpack.dct(blockedImg, axis=-1, norm="ortho")
	blockedImg = scipy.fftpack.dct(blockedImg, axis=-2, norm="ortho")
	return blockedImg

#
# Apply masking coefficients to DCT form of images.
#

def maskeff(M, M_dct, eps=1e-10):
	m   =  np.sum((M_dct**2) * MaskC0f, axis=(-2,-1), keepdims=True)
	pop = (np.var(M,            axis=(-2,-1), keepdims=True, ddof=1) *
	       np.prod(M.shape[-2:]) + eps)
	var = (np.var(M[...,:4,:4], axis=(-2,-1), keepdims=True, ddof=1) +
	       np.var(M[...,4:,:4], axis=(-2,-1), keepdims=True, ddof=1) +
	       np.var(M[...,:4,4:], axis=(-2,-1), keepdims=True, ddof=1) +
	       np.var(M[...,4:,4:], axis=(-2,-1), keepdims=True, ddof=1)) * 16.0
	pop = np.where(np.abs(pop)<1, pop, (var/pop))
	return np.sqrt(m*pop)/32


if __name__ == "__main__":
	import sys
	img1 = cv.imread(sys.argv[1])
	img2 = cv.imread(sys.argv[2])
	print("{:.20f}".format(psnrhma_color(img1, img2)))
