# -*- coding: utf-8 -*-
import numpy as np


"""
Illuminant White Points for CIE 1931 2° standard observer.

Chromaticities given in x,y coordinates.

Reference:
    ITU-T Rec. H.264 (10/2016)
    https://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants
"""
WHITE_A   = (0.44758, 0.40745)
WHITE_C   = (0.310,   0.316  )
WHITE_D50 = (0.34567, 0.35850)
WHITE_D65 = (0.3127,  0.3290 )
WHITE_E   = (1./3,    1./3   )


"""
Color Primaries.

Chromaticities given in x,y coordinates.

Reference:
    ITU-T Rec. H.264 (10/2016)
"""
PRIMARY_R_sRGB          = (0.640, 0.330)
PRIMARY_G_sRGB          = (0.300, 0.600)
PRIMARY_B_sRGB          = (0.150, 0.060)
PRIMARY_R_BT470M        = (0.670, 0.330)
PRIMARY_G_BT470M        = (0.210, 0.710)
PRIMARY_B_BT470M        = (0.140, 0.080)
PRIMARY_R_BT470BG       = (0.640, 0.330)
PRIMARY_G_BT470BG       = (0.290, 0.600)
PRIMARY_B_BT470BG       = (0.150, 0.060)
PRIMARY_R_SMPTE170M     = (0.630, 0.340)
PRIMARY_G_SMPTE170M     = (0.310, 0.595)
PRIMARY_B_SMPTE170M     = (0.155, 0.070)
PRIMARY_R_SMPTE240M     = PRIMARY_R_SMPTE170M
PRIMARY_G_SMPTE240M     = PRIMARY_G_SMPTE170M
PRIMARY_B_SMPTE240M     = PRIMARY_B_SMPTE170M
PRIMARY_R_FILM          = (0.681, 0.319)
PRIMARY_G_FILM          = (0.243, 0.692)
PRIMARY_B_FILM          = (0.145, 0.049)
PRIMARY_R_BT2020        = (0.708, 0.292)
PRIMARY_G_BT2020        = (0.170, 0.797)
PRIMARY_B_BT2020        = (0.131, 0.046)
PRIMARY_R_BT2100        = PRIMARY_R_BT2020
PRIMARY_G_BT2100        = PRIMARY_G_BT2020
PRIMARY_B_BT2100        = PRIMARY_B_BT2020
PRIMARY_R_SMPTE_RP431_2 = (0.680, 0.320)
PRIMARY_G_SMPTE_RP431_2 = (0.265, 0.690)
PRIMARY_B_SMPTE_RP431_2 = (0.150, 0.060)
PRIMARY_R_SMPTE_RP432_1 = PRIMARY_R_SMPTE_RP431_2
PRIMARY_G_SMPTE_RP432_1 = PRIMARY_G_SMPTE_RP431_2
PRIMARY_B_SMPTE_RP432_1 = PRIMARY_B_SMPTE_RP431_2
PRIMARY_R_EBU3213E      = (0.630, 0.340)
PRIMARY_G_EBU3213E      = (0.295, 0.605)
PRIMARY_B_EBU3213E      = (0.155, 0.077)


"""
White point chromaticity of D-series illuminants

Reference:
    https://en.wikipedia.org/wiki/Standard_illuminant#Illuminant_series_D
"""
def getDWhitePoint(T=6500):
	assert T >= 4000 and T <= 25000
	if T >= 4000 and T <= 7000: p = [-4.6070e9, +2.9678e6, +0.09911e3, +0.244063]
	else:                       p = [-2.0064e9, +1.9018e6, +0.24748e3, +0.237040]
	x = np.polyval(p, 1./T)
	y = np.polyval([-3.000, +2.870, -0.275], x)
	return (x,y)


"""
XYZ <-> linear RGB matrix calculations
"""
def getRGBtoXYZMatrix(r = PRIMARY_R_sRGB,
                      g = PRIMARY_G_sRGB,
                      b = PRIMARY_B_sRGB,
                      w = WHITE_D65):
	M  = np.asarray([[r[0],         g[0],         b[0]],
	                 [r[1],         g[1],         b[1]],
	                 [1.-r[0]-r[1], 1.-g[0]-g[1], 1.-b[0]-b[1]]], dtype=np.float64)
	W  = np.asarray([w[0], w[1], 1.0-w[0]-w[1]], dtype=np.float64)/w[1]
	S  = np.linalg.solve(M,W)
	M *= S
	return M

def getXYZtoRGBMatrix(r = PRIMARY_R_sRGB,
                      g = PRIMARY_G_sRGB,
                      b = PRIMARY_B_sRGB,
                      w = WHITE_D65):
	return np.linalg.inv(getRGBtoXYZMatrix(r,g,b,w))


"""
Color Adaptation in XYZ space

Reference:
    http://www.brucelindbloom.com/index.html?Eqn_DIlluminant.html
    https://pdfs.semanticscholar.org/f082/4a43c8b21a357266159a55605daf5e8a7cc3.pdf
"""
def getCATMatrix(wd=WHITE_D50, ws=WHITE_D65):
	MA  = np.asarray([[+0.8951000, +0.2664000, -0.1614000],
	                  [-0.7502000, +1.7135000, +0.0367000],
	                  [+0.0389000, -0.0685000, +1.0296000]], dtype=np.float64)
	Wd  = np.asarray([wd[0], wd[1], 1.-wd[0]-wd[1]], dtype=np.float64)/wd[1]
	Ws  = np.asarray([ws[0], ws[1], 1.-ws[0]-ws[1]], dtype=np.float64)/ws[1]
	WAd = MA.dot(Wd)
	WAs = MA.dot(Ws)
	D   = np.diag(WAd/WAs)
	return np.linalg.inv(MA).dot(D).dot(MA)

