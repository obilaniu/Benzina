/* Include Guard */
#ifndef INCLUDE_BENZINA_BITS_H
#define INCLUDE_BENZINA_BITS_H



/* Defines */

/**
 * Color Primaries
 * 
 * The red, green and blue primaries, plus the white point, expressed in CIE xy
 * chromaticity coordinates.
 * 
 * References:
 *     ITU-T H.264      2017-04
 *     ITU-T H.265      2018-02
 *     ITU-T H.273      2016-12
 * 
 * 
 *                                    Code   |      R       |      G       |      B       |         White        | Notes...
 *                                           |     x,y      |     x,y      |     x,y      |      x,y       Desc. |
 */
#define BENZINA_PRI_RESERVED0           0  //|              |              |              |                      | Reserved
#define BENZINA_PRI_BT_709              1  //| 0.640, 0.330 | 0.300, 0.600 | 0.150, 0.060 | 0.3127, 0.3290 (D65) | SMPTE RP 177 Annex B (1993), BT.1361-0 conventional & extended colour gamut, IEC 61966-2-1 sRGB/sYCC, IEC 61966-2-4
#define BENZINA_PRI_UNSPECIFIED         2  //|              |              |              |                      | Unknown or determined by application
#define BENZINA_PRI_RESERVED            3  //|              |              |              |                      | Reserved
#define BENZINA_PRI_BT_470M             4  //| 0.670, 0.330 | 0.210, 0.710 | 0.140, 0.080 | 0.3100, 0.3160 (C)   | NTSC 1953, US FCC Title 47 (2003) 73.682 (a)(20)
#define BENZINA_PRI_BT_470BG            5  //| 0.640, 0.330 | 0.290, 0.600 | 0.150, 0.060 | 0.3127, 0.3290 (D65) | BT.601-7 625 lines, BT.1358-0 625 lines, BT.1700-0 625 lines PAL & 625 lines SECAM
#define BENZINA_PRI_SMPTE_ST_170M       6  //| 0.630, 0.340 | 0.310, 0.595 | 0.155, 0.070 | 0.3127, 0.3290 (D65) | SMPTE ST 170M (2004), BT.601-7 525 lines, BT.1358-1 525 or 625 lines, BT.1700-0 NTSC
#define BENZINA_PRI_SMPTE_ST_240M       7  //| 0.630, 0.340 | 0.310, 0.595 | 0.155, 0.070 | 0.3127, 0.3290 (D65) | SMPTE ST 240M (1999)
#define BENZINA_PRI_FILM                8  //| 0.681, 0.319 | 0.243, 0.692 | 0.145, 0.049 | 0.3100, 0.3160 (C)   | Generic film using Illuminant C.
#define BENZINA_PRI_BT_2020             9  //| 0.708, 0.292 | 0.170, 0.797 | 0.131, 0.046 | 0.3127, 0.3290 (D65) | BT.2020-2, BT.2100-0
#define BENZINA_PRI_SMPTE_ST_428       10  //| 1.000, 0.000 | 0.000, 1.000 | 0.000, 0.000 |  1./3,   1./3  (E)   | SMPTE ST 428-1 (2006), CIE 1931 XYZ as in ISO 11664-1
#define BENZINA_PRI_SMPTE_RP_431       11  //| 0.680, 0.320 | 0.265, 0.690 | 0.150, 0.060 | 0.3140, 0.3510       | SMPTE RP 431-2 (2011)
#define BENZINA_PRI_SMPTE_EG_432       12  //| 0.680, 0.320 | 0.265, 0.690 | 0.150, 0.060 | 0.3127, 0.3290 (D65) | SMPTE EG 432-1 (2010)
#define BENZINA_PRI_EBU_3213E          22  //| 0.630, 0.340 | 0.295, 0.605 | 0.155, 0.077 | 0.3127, 0.3290 (D65) | EBU Tech. 3213-E (1975)



/**
 * Transfer Characteristic
 * 
 * Defines an OETF (Opto-Electronic Transfer Function) or inverse EOTF
 * (Electro-Optical Transfer Function). There are certain points to be taken
 * into account, however:
 * 
 *   - 1,6,14,15: ITU-R standards have defined reference OETFs, but never explicitly defined a reference EOTF for CRTs.
 *                To solve this, ITU-R BT.1886-0 defined a possible reference EOTF power law for CRTs.
 *   - 4:         No explicit OETF, but EOTF assumed to be L = pow(V, 2.2)
 *   - 5:         No explicit OETF, but EOTF assumed to be L = pow(V, 2.8)
 *   - 16:        ITU-R BT.2100 PQ  defines a reference EOTF, but it is not the same as the inverse OETF.
 *   - 18:        ITU-R BT.2100 HLG defines a reference OETF, but it is not the same as the inverse EOTF.
 * 
 * 
 * References:
 *     ITU-T H.264      2017-04
 *     ITU-T H.265      2018-02
 *     ITU-T H.273      2016-12
 *     ITU-T H.Supp15   2017-01
 *     ITU-T H.Supp18   2017-10
 *     ITU-R BT.470-6   1998-11
 *     ITU-R BT.601-7   2011-03
 *     ITU-R BT.1358-0  1998-02
 *     ITU-R BT.1358-1  2007-09
 *     ITU-R BT.1361-0  1998-02
 *     ITU-R BT.1700-0  2005-02
 *     ITU-R BT.1886-0  2011-03
 *     ITU-R BT.2020-0  2012-08
 *     ITU-R BT.2020-1  2014-06
 *     ITU-R BT.2020-2  2015-10
 *     ITU-R BT.2100-0  2016-07
 *     ITU-R BT.2100-1  2017-06
 *     ITU-R BT.2100-2  2018-07
 *     ITU-R BT.2390-0  2016-02
 *     ITU-R BT.2390-1  2016-10
 *     ITU-R BT.2390-2  2017-03
 *     ITU-R BT.2390-3  2017-10
 *     ITU-R BT.2390-4  2018-04
 *     ITU-R BT.2390-5  2018-10
 * 
 *                                    Code   | Notes...
 */
#define BENZINA_XFR_RESERVED0           0  //| Reserved
#define BENZINA_XFR_BT_709              1  //| BT.1361-0 conventional colour gamut. Functionally identical to 1,6,14,15.
#define BENZINA_XFR_UNSPECIFIED         2  //| Unknown or determined by application
#define BENZINA_XFR_RESERVED            3  //| Reserved
#define BENZINA_XFR_GAMMA22             4  //| Assumed display gamma 2.2. BT.470-6 M, NTSC 1953, US FCC Title 47 (2003) 73.682 (a)(20)
#define BENZINA_XFR_GAMMA28             5  //| Assumed display gamma 2.8. BT.470-6 BG, BT.1700-0 625 lines PAL & 625 lines SECAM
#define BENZINA_XFR_SMPTE_ST_170M       6  //| SMPTE ST 170M (2004), BT.601-7 525 or 625 lines, BT.1358-1 525 or 625 lines, BT.1700-0 NTSC. Functionally identical to 1,6,14,15.
#define BENZINA_XFR_SMPTE_ST_240M       7  //| SMPTE ST 240M (1999).
#define BENZINA_XFR_LINEAR              8  //| Linear
#define BENZINA_XFR_LOG20               9  //| Logarithmic, 10**2.0 : 1 range.
#define BENZINA_XFR_LOG25              10  //| Logarithmic, 10**2.5 : 1 range.
#define BENZINA_XFR_IEC_61966_2_4      11  //| IEC 61966-2-4
#define BENZINA_XFR_BT_1361            12  //| BT.1361-0 extended colour gamut
#define BENZINA_XFR_IEC_61966_2_1      13  //| IEC 61966-2-1 sRGB/sYCC
#define BENZINA_XFR_BT_2020_10BIT      14  //| BT.2020-2 10-bit system. Functionally identical to 1,6,14,15.
#define BENZINA_XFR_BT_2020_12BIT      15  //| BT.2020-2 12-bit system. Functionally identical to 1,6,14,15.
#define BENZINA_XFR_BT_2100_PQ         16  //| BT.2100-1 PQ, SMPTE ST 2084 (2014) for 10, 12, 14 and 16-bit systems.
#define BENZINA_XFR_SMPTE_ST_428       17  //| SMPTE ST 428-1 (2006)
#define BENZINA_XFR_BT_2100_HLG        18  //| BT.2100-1 HLG, ARIB STD-B67



/**
 * Color Matrix
 * 
 * The color matrix that converts from non-linear RGB to the final colorspace.
 * 
 * References:
 *     ITU-T H.264      2017-04
 *     ITU-T H.265      2018-02
 *     ITU-T H.273      2016-12
 * 
 * 
 *                                    Code   |     Kr,Kg      | Notes...
 */
#define BENZINA_CSC_IDENTITY            0  //|                | Identity matrix (GBR/YZX). IEC 61966-2-1 sRGB, SMPTE ST 428-1
#define BENZINA_CSC_BT_709              1  //| 0.2126, 0.0722 | BT.709-6, BT.1361-0 conventional & extended colour gamut, IEC 61966-2-1 sYCC, IEC 61966-2-4 xvYCC709, SMPTE RP 177 Annex B (1993)
#define BENZINA_CSC_UNSPECIFIED         2  //|                | Unknown or determined by application
#define BENZINA_CSC_RESERVED            3  //|                | Reserved
#define BENZINA_CSC_FCC                 4  //| 0.3000, 0.1100 | US FCC Title 47 (2003) 73.682 (a)(20)
#define BENZINA_CSC_BT_470BG            5  //| 0.2990, 0.1140 | BT.470-6 BG, BT.601-7 625 lines, BT.1358-0 625 lines, BT.1700-0 625 lines PAL & 625 lines SECAM, IEC 61966-2-4 xvYCC601. Functionally identical to 6.
#define BENZINA_CSC_SMPTE_ST_170M       6  //| 0.2990, 0.1140 | SMPTE ST 170M (2004), BT.601-7 525 or 625 lines, BT.1358-1 525 or 625 lines, BT.1700-0 NTSC. Functionally identical to 5.
#define BENZINA_CSC_SMPTE_ST_240M       7  //| 0.2120, 0.0870 | SMPTE ST 240M (1999).
#define BENZINA_CSC_YCGCO               8  //|                | YCgCo
#define BENZINA_CSC_BT_2020_NCL         9  //| 0.2627, 0.0593 | BT.2020-2 NCL, BT.2100-1 YCbCr
#define BENZINA_CSC_BT_2020_CL         10  //| 0.2627, 0.0593 | BT.2020-2 CL
#define BENZINA_CSC_YDZDX              11  //|                | Y'D'zD'x
#define BENZINA_CSC_CHROMATICITY_NCL   12  //|    (derived)   | Chromaticity-derived NCL
#define BENZINA_CSC_CHROMATICITY_CL    13  //|    (derived)   | Chromaticity-derived CL
#define BENZINA_CSC_ICTCP              14  //|                | ICtCp



/**
 * Chroma Format
 * 
 * Determines the chroma subsampling.
 * 
 * References:
 *     ITU-T H.264      2017-04
 *     ITU-T H.265      2018-02
 * 
 *                                    Code   | Notes...
 */
#define BENZINA_CHROMAFMT_YUV400        0  //| YUV 4:0:0 (monochrome)
#define BENZINA_CHROMAFMT_YUV420        1  //| YUV 4:2:0 (2x vertical, 2x horizontal chroma subsampling)
#define BENZINA_CHROMAFMT_YUV422        2  //| YUV 4:2:2 (2x horizontal chroma subsampling)
#define BENZINA_CHROMAFMT_YUV444        3  //| YUV 4:4:4 (no subsampling)



/**
 * Chroma Sample Location
 * 
 * Determines where the top-left chroma sample of a YUV 4:2:0-chroma-subsampled
 * array is located, relative to the top-level luma sample.
 * 
 * In ASCII art, the options (0-5) relative to the top-left luma sample (X) are:
 * 
 *          Chroma             Luma
 *          X   .   X          2   3   .
 *
 *          .   .   .          0   1   .
 *
 *          X   .   X          4   5   .
 * 
 * 
 * References:
 *     ITU-T H.263      2005-01
 *     ITU-T H.264      2017-04
 *     ITU-T H.265      2018-02
 * 
 *                                    Code   | Notes...
 */
#define BENZINA_CHROMALOC_LEFT          0  //| Left, between left-most luma samples. MPEG-2/MPEG-4/H.264/H.265 for 4:2:0
#define BENZINA_CHROMALOC_CENTER        1  //| Center, between four luma samples. MPEG-1/H.263/JPEG for 4:2:0
#define BENZINA_CHROMALOC_TOPLEFT       2  //| Top-left, co-sited with top-left luma sample. ITU-R BT.601 for 4:2:2
#define BENZINA_CHROMALOC_TOP           3  //| Top, between top-most luma samples.
#define BENZINA_CHROMALOC_BOTTOMLEFT    4  //| Bottom-left, co-sited with left-most luma sample of second row. Extremely rare.
#define BENZINA_CHROMALOC_BOTTOM        5  //| Bottom, between left-most luma samples of second row. Extremely rare.
#define BENZINA_CHROMALOC_TOPRIGHT      6  //| NON-STANDARD: Top-right, co-sited with second luma sample of first row.
#define BENZINA_CHROMALOC_RIGHT         7  //| NON-STANDARD: Center-right, between second-left-most luma samples.
#define BENZINA_CHROMALOC_BOTTOMRIGHT   8  //| NON-STANDARD: Bottom-right, co-sited with second luma sample of second row.



/**
 * Color Range/Swing
 * 
 * YCbCr data can be coded with a "restricted" range that includes headroom and footroom
 * above, or in "full" range.
 * 
 * Restricted range was once upon a time necessary to give analog filters head/footroom
 * to avoid clipping, but it results in a small loss that is completely unnecessary in
 * digital systems.
 * 
 *                                    Code   | Notes...
 */
#define BENZINA_SWING_REDUCED           0  //| Reduced-range:
#define BENZINA_SWING_MPEG              0  //|   0..2^n-1
#define BENZINA_SWING_TV                0  //| where n is the bit-depth.
#define BENZINA_SWING_ANALOG            0  //|
#define BENZINA_SWING_FULL              1  //| Full-range:
#define BENZINA_SWING_JPEG              1  //|   0..219*2^(n-8)
#define BENZINA_SWING_PC                1  //| where n is the bit-depth.
#define BENZINA_SWING_DIGITAL           1  //|



/* End Include Guard */
#endif

