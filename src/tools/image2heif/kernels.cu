/**
 * Collection of CUDA kernels that accurately resamples and convert from
 * FFmpeg's better-supported pixel formats to full-range YUV420P.
 * 
 * 
 * Not supported:
 *   - Bayer Pattern:    bayer_*
 *   - Sub-8-bit RGB:    bgr4, rgb4, bgr8, rgb8, bgr4_byte, rgb4_byte,
 *                       bgr444be, bgr444le, rgb444be, rgb444le,
 *                       bgr555be, bgr555le, rgb555be, rgb555le,
 *                       bgr565be, bgr565le, rgb565be, rgb565le
 *   - Hardware formats: ...
 *   - XYZ:              xyz12be, xyz12le
 */

/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "kernels.h"



/* Defines */

/**
 * Kernel launch bounds.
 * 
 * In all Compute Capability architectures <7.5, the max threads per SM is
 * 2048 and the max threads per block is 1024, meaning that 2 blocks of 1024
 * had always been fine. In 7.5 (Turing), this changed and now an SM can fit
 * at most 1024 threads.
 */

#ifdef __CUDA_ARCH__
# if   __CUDA_ARCH__ < 750
#  define KERNEL_BOUNDS __launch_bounds__(1024, 2)
# else
#  define KERNEL_BOUNDS __launch_bounds__(1024, 1)
# endif
#else
# define KERNEL_BOUNDS
#endif

/**
 * Transfer Function codes
 * 
 * Reference:
 *     ITU-T Rec. H.264 (10/2016)
 */

#define TRC_BT709_5         1
#define TRC_UNSPECIFIED     2
#define TRC_GAMMA_2_2       4
#define TRC_GAMMA_2_8       5
#define TRC_SMPTE170M       6
#define TRC_SMPTE240M       7
#define TRC_LINEAR          8
#define TRC_LOG10_2         9
#define TRC_LOG10_2_5      10
#define TRC_IEC61966_2_4   11
#define TRC_BT1361_0       12
#define TRC_IEC61966_2_1   13
#define TRC_BT2020_2_10    14
#define TRC_BT2020_2_12    15
#define TRC_BT2100PQ       16
#define TRC_SMPTE_ST428_1  17



/* CUDA kernels */

/**
 * @brief Floating-point clamp
 */

static __host__ __device__ float  fclampf(float lo, float x, float hi){
    return fmaxf(lo, fminf(x, hi));
}

/**
 * @brief Elementwise math Functions
 */

static __host__ __device__ float3 lerps3(float3 a, float3 b, float m){
    return make_float3((1.0f-m)*a.x + (m)*b.x,
                       (1.0f-m)*a.y + (m)*b.y,
                       (1.0f-m)*a.z + (m)*b.z);
}
static __host__ __device__ float3 lerpv3(float3 a, float3 b, float3 m){
    return make_float3((1.0f-m.x)*a.x + (m.x)*b.x,
                       (1.0f-m.y)*a.y + (m.y)*b.y,
                       (1.0f-m.z)*a.z + (m.z)*b.z);
}

/**
 * @brief Forward/Backward Transfer Functions
 * 
 * Some of the backward functions are not yet implemented, because
 * deriving them is non-trivial.
 * 
 * Reference:
 *     ITU-T Rec. H.264   (10/2016)
 *     http://avisynth.nl/index.php/Colorimetry
 */

static __host__ __device__ float  fwXfrFn (float  L, unsigned fn){
    float a, b, g, c1, c2, c3, n, m, p;
    
    switch(fn){
        case TRC_BT709_5:
        case TRC_SMPTE170M:
        case TRC_BT2020_2_10:
        case TRC_BT2020_2_12:
            a = 1.099296826809442f;
            b = 0.018053968510807f;
            return L >= b ? a*powf(L, 0.45f) - (a-1.f) : 4.500f*L;
        case TRC_GAMMA_2_2: return powf(fmaxf(L, 0.f), 1.f/2.2f);
        case TRC_GAMMA_2_8: return powf(fmaxf(L, 0.f), 1.f/2.8f);
        case TRC_SMPTE240M:
            a = 1.111572195921731f;
            b = 0.022821585529445f;
            return L >= b ? a*powf(L, 0.45f) - (a-1.f) : 4.0f*L;
        case TRC_LINEAR:       return L;
        case TRC_LOG10_2:      return L < sqrtf(1e-4) ? 0.f : 1.f + log10f(L)/2.0f;
        case TRC_LOG10_2_5:    return L < sqrtf(1e-5) ? 0.f : 1.f + log10f(L)/2.5f;
        case TRC_IEC61966_2_4:
            a = 1.099296826809442f;
            b = 0.018053968510807f;
            if      (L >= +b){return +a*powf(+L, 0.45f) - (a-1.f);
            }else if(L <= -b){return -a*powf(-L, 0.45f) + (a-1.f);
            }else{            return 4.5f*L;
            }
        case TRC_BT1361_0:
            a = 1.099296826809442f;
            b = 0.018053968510807f;
            g = b/4.f;
            if      (L >= +b){return +(a*powf(+1.f*L, 0.45f) - (a-1.f));
            }else if(L <= -g){return -(a*powf(-4.f*L, 0.45f) - (a-1.f))/4.f;
            }else{            return 4.5f*L;
            }
        case TRC_UNSPECIFIED:
        case TRC_IEC61966_2_1:
        default:
            a = 1.055f;
            b = 0.0031308f;
            return L > b ? a*powf(L, 1.f/2.4f) - (a-1.f) : 12.92f*L;
        case TRC_BT2100PQ:
            c1 =  107.f/  128.f;
            c2 = 2413.f/  128.f;
            c3 = 2392.f/  128.f;
            m  = 2523.f/   32.f;
            n  = 2610.f/16384.f;
            p  = powf(fmaxf(L, 0.f), n);
            return powf((c1+c2*p)/(1.f+c3*p), m);
        case TRC_SMPTE_ST428_1: return powf(48.f*fmaxf(L, 0.f)/52.37f, 1.f/2.6f);
    }
}
static __host__ __device__ float  bwXfrFn (float  V, unsigned fn){
    float a, b;
    
    switch(fn){
        case TRC_BT709_5:
        case TRC_SMPTE170M:
        case TRC_BT2020_2_10:
        case TRC_BT2020_2_12:
            a = 0.099296826809442f;
            b = 0.08124285829863229f;
            return V >= b ? powf((V+a)/(1.f+a), 1.f/0.45f) : V/4.500f;
        case TRC_GAMMA_2_2: return powf(V, 2.2f);
        case TRC_GAMMA_2_8: return powf(V, 2.8f);
        case TRC_SMPTE240M:
            a = 1.111572195921731f;
            b = 0.09128634211778018f;
            return V >= b ? powf((V+a)/(1.f+a), 1.f/0.45f) : V/4.0f;
        case TRC_LINEAR:        return V;
        case TRC_LOG10_2:       return powf(10.f, (V-1.f)*2.0f);
        case TRC_LOG10_2_5:     return powf(10.f, (V-1.f)*2.5f);
        case TRC_IEC61966_2_4:  return 0.0f;/* FIXME: Implement me! */
        case TRC_BT1361_0:      return 0.0f;/* FIXME: Implement me! */
        case TRC_UNSPECIFIED:
        case TRC_IEC61966_2_1:
        default:
            a = 0.055f;
            b = 0.04045f;
            return V >  b ? powf((V+a)/(1.f+a),      2.4f) : V/12.92f;
        case TRC_BT2100PQ:      return 0.0f;/* FIXME: Implement me! */
        case TRC_SMPTE_ST428_1: return powf(V, 2.6f)*52.37f/48.f;
    }
}
static __host__ __device__ float3 fwXfrCol(float3   p,
                                           unsigned cm,
                                           unsigned fn,
                                           unsigned fullRange,
                                           unsigned YDepth,
                                           unsigned CDepth){
    float Eg=p.x, Eb=p.y, Er=p.z;
    float Ey;
    float Epy, Eppb, Eppr;
    float Epr, Epg,  Epb;
    float Kr, Kb;
    float Y,  Cb, Cr;
    float R,  G,  B;
    float Nb, Pb, Nr, Pr;
    float YClamp, YSwing, YShift;
    float CClamp, CSwing, CShift;
    
    if(fullRange){
        switch(cm){
            case 0:
            case 8:
                YClamp=(1<<YDepth)-1;
                YSwing=(1<<YDepth)-1;
                YShift=0;
                CClamp=(1<<CDepth)-1;
                CSwing=(1<<CDepth)-1;
                CShift=0;
            break;
            default:
                YClamp=(1<<YDepth)-1;
                YSwing=(1<<YDepth)-1;
                YShift=0;
                CClamp=(1<<CDepth)-1;
                CSwing=(1<<CDepth)-1;
                CShift=(1<<(CDepth-1));
            break;
        }
    }else{
        switch(cm){
            case 0:
            case 8:
                YClamp=(1<<YDepth)-1;
                YSwing=219<<(YDepth-8);
                YShift= 16<<(YDepth-8);
                CClamp=(1<<CDepth)-1;
                CSwing=219<<(CDepth-8);
                CShift= 16<<(CDepth-8);
            break;
            default:
                YClamp=(1<<YDepth)-1;
                YSwing=219<<(YDepth-8);
                YShift= 16<<(YDepth-8);
                CClamp=(1<<CDepth)-1;
                CSwing=224<<(CDepth-8);
                CShift=(1<<(CDepth-1));
            break;
        }
    }
    switch(cm){
        case  1: Kr=0.2126, Kb=0.0722; break;
        case  4: Kr=0.30,   Kb=0.11;   break;
        case  5:
        case  6:
        default: Kr=0.299,  Kb=0.114;  break;
        case  7: Kr=0.212,  Kb=0.087;  break;
        case  9:
        case 10: Kr=0.2627, Kb=0.0593; break;
    }
    switch(cm){
        case  2:
            /* Unknown. Fall-through to common case. */
        case  1:
        case  4:
        case  5:
        case  6:
        case  7:
        case  9:
        default:
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epg  = fwXfrFn(Eg, fn);                                    /* E-2. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            Epy  = Kr*Epr + (1-Kr-Kb)*Epg + Kb*Epb;                    /* E-16 */
            Eppb = 0.5f*(Epb-Epy)/(1-Kb);                              /* E-17 */
            Eppr = 0.5f*(Epr-Epy)/(1-Kr);                              /* E-18 */
            Y    = fclampf(0, YSwing*Epy  + YShift, YClamp);           /* E-4 */
            Cb   = fclampf(0, CSwing*Eppb + CShift, CClamp);           /* E-5 */
            Cr   = fclampf(0, CSwing*Eppr + CShift, CClamp);           /* E-6 */
        break;
        case 10:
            Nb   = fwXfrFn(1-Kb, fn);                                  /* E-43 */
            Pb   = 1-fwXfrFn(Kb, fn);                                  /* E-44 */
            Nr   = fwXfrFn(1-Kr, fn);                                  /* E-45 */
            Pr   = 1-fwXfrFn(Kr, fn);                                  /* E-46 */
            Ey   = Kr*Er + (1-Kr-Kb)*Eg + Kb*Eb;                       /* E-37 */
            Epy  = fwXfrFn(Ey, fn);                                    /* E-38 */
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            Eppb = (Epb-Epy)<0 ? (Epb-Epy)/(2*Nb) : (Epb-Epy)/(2*Pb);  /* E-39, E-40 */
            Eppr = (Epr-Epy)<0 ? (Epr-Epy)/(2*Nr) : (Epr-Epy)/(2*Pr);  /* E-41, E-42 */
            Y    = fclampf(0, YSwing*Epy  + YShift, YClamp);           /* E-4 */
            Cb   = fclampf(0, CSwing*Eppb + CShift, CClamp);           /* E-5 */
            Cr   = fclampf(0, CSwing*Eppr + CShift, CClamp);           /* E-6 */
        break;
        case 11:
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epg  = fwXfrFn(Eg, fn);                                    /* E-2. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            Epy  = Epg;                                                /* E-47. Y'  */
            Eppb = (0.986566f*Epb - Epy) * 0.5f;                       /* E-48. D'z */
            Eppr = (Epr - 0.991902f*Epy) * 0.5f;                       /* E-49. D'x */
            Y    = fclampf(0, YSwing*Epy  + YShift, YClamp);           /* E-4 */
            Cb   = fclampf(0, CSwing*Eppb + CShift, CClamp);           /* E-5 */
            Cr   = fclampf(0, CSwing*Eppr + CShift, CClamp);           /* E-6 */
        break;
        case  0:
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epg  = fwXfrFn(Eg, fn);                                    /* E-2. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            R    = fclampf(0, YSwing*Epr + YShift, YClamp);            /* E-7 */
            G    = fclampf(0, CSwing*Epg + CShift, CClamp);            /* E-8 */
            B    = fclampf(0, CSwing*Epb + CShift, CClamp);            /* E-9 */
            Y    = G;                                                  /* E-19 */
            Cb   = B;                                                  /* E-20 */
            Cr   = R;                                                  /* E-21 */
        break;
        case  8:
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epg  = fwXfrFn(Eg, fn);                                    /* E-2. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            R    = fclampf(0, YSwing*Epr + YShift, YClamp);            /* E-7 */
            G    = fclampf(0, YSwing*Epg + YShift, CClamp);            /* E-8 */
            B    = fclampf(0, YSwing*Epb + YShift, CClamp);            /* E-9 */
            Y    = 0.5f*G + 0.25f*(R+B);                               /* E-22. Y  */
            Cb   = 0.5f*G - 0.25f*(R+B) + (1<<(CDepth-1));             /* E-23. Cg */
            Cr   = 0.5f*(R-B)           + (1<<(CDepth-1));             /* E-24. Co */
        break;
    }
    
    return make_float3(Y, Cb, Cr);
}
static __host__ __device__ float3 bwXfrCol(float3 p,
                                           unsigned cm,
                                           unsigned fn,
                                           unsigned fullRange,
                                           unsigned YDepth,
                                           unsigned CDepth){
    float Y=p.x,  Cb=p.y, Cr=p.z;
    float Epy, Eppb, Eppr;
    float Epr, Epg,  Epb;
    float Er,  Eg,   Eb;
    float Ey;
    float Kr, Kb;
    float R,  G,  B,  t;
    float Nb, Pb, Nr, Pr;
    float YSwing, YShift;
    float CSwing, CShift;
    
    if(fullRange){
        switch(cm){
            case 0:
            case 8:
                YSwing=(1<<YDepth)-1;
                YShift=0;
                CSwing=(1<<CDepth)-1;
                CShift=0;
            break;
            default:
                YSwing=(1<<YDepth)-1;
                YShift=0;
                CSwing=(1<<CDepth)-1;
                CShift=(1<<(CDepth-1));
            break;
        }
    }else{
        switch(cm){
            case 0:
            case 8:
                YSwing=219<<(YDepth-8);
                YShift= 16<<(YDepth-8);
                CSwing=219<<(CDepth-8);
                CShift= 16<<(CDepth-8);
            break;
            default:
                YSwing=219<<(YDepth-8);
                YShift= 16<<(YDepth-8);
                CSwing=224<<(CDepth-8);
                CShift=(1<<(CDepth-1));
            break;
        }
    }
    switch(cm){
        case  1: Kr=0.2126, Kb=0.0722; break;
        case  4: Kr=0.30,   Kb=0.11;   break;
        case  5:
        case  6:
        default: Kr=0.299,  Kb=0.114;  break;
        case  7: Kr=0.212,  Kb=0.087;  break;
        case  9:
        case 10: Kr=0.2627, Kb=0.0593; break;
    }
    switch(cm){
        case  2:
            /* Unknown. Fall-through to common case. */
        case  1:
        case  4:
        case  5:
        case  6:
        case  7:
        case  9:
        default:
            Epy  = fclampf(0, (Y -YShift)/YSwing, 1);                  /* E-4 */
            Eppb = fclampf(0, (Cb-CShift)/CSwing, 1);                  /* E-5 */
            Eppr = fclampf(0, (Cr-CShift)/CSwing, 1);                  /* E-6 */
            Epr  = Epy + (1-Kr)*2*Eppr;                                /* E-16,17,18 Inverse */
            Epg  = Epy - (1-Kb)*Kb/(1-Kb-Kr)*2*Eppb                    /* E-16,17,18 Inverse */
                       - (1-Kr)*Kr/(1-Kb-Kr)*2*Eppr;                   /* E-16,17,18 Inverse */
            Epb  = Epy + (1-Kb)*2*Eppb;                                /* E-16,17,18 Inverse */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eg   = bwXfrFn(Epg, fn);                                   /* E-2. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
        break;
        case 10:
            Nb   = fwXfrFn(1-Kb, fn);                                  /* E-43 */
            Pb   = 1-fwXfrFn(Kb, fn);                                  /* E-44 */
            Nr   = fwXfrFn(1-Kr, fn);                                  /* E-45 */
            Pr   = 1-fwXfrFn(Kr, fn);                                  /* E-46 */
            Epy  = fclampf(0, (Y -YShift)/YSwing, 1);                  /* E-4 */
            Eppb = fclampf(0, (Cb-CShift)/CSwing, 1);                  /* E-5 */
            Eppr = fclampf(0, (Cr-CShift)/CSwing, 1);                  /* E-6 */
            Epb  = Eppb<0 ? Eppb*2*Nb+Epy : Eppb*2*Pb+Epy;             /* E-39, E-40 */
            Epr  = Eppr<0 ? Eppr*2*Nr+Epy : Eppr*2*Pr+Epy;             /* E-39, E-40 */
            Ey   = bwXfrFn(Epy, fn);                                   /* E-38 */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
            Eg   = (Ey - Kr*Er - Kb*Eb)/(1-Kr-Kb);                     /* E-37 */
        break;
        case 11:
            Epy  = fclampf(0, (Y -YShift)/YSwing, 1);                  /* E-4 */
            Eppb = fclampf(0, (Cb-CShift)/CSwing, 1);                  /* E-5 */
            Eppr = fclampf(0, (Cr-CShift)/CSwing, 1);                  /* E-6 */
            Epg  = Epy;                                                /* E-47. Y'  */
            Epb  = (2*Eppb + Epg)/0.986566f;                           /* E-48. D'z */
            Epr  = 2*Eppr + 0.991902f*Epg;                             /* E-49. D'x */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eg   = bwXfrFn(Epg, fn);                                   /* E-2. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
        break;
        case  0:
            G    = Y;                                                  /* E-19 */
            B    = Cb;                                                 /* E-20 */
            R    = Cr;                                                 /* E-21 */
            Epr  = fclampf(0, (R-YShift)/YSwing, 1);                   /* E-7 */
            Epg  = fclampf(0, (G-YShift)/YSwing, 1);                   /* E-8 */
            Epb  = fclampf(0, (B-YShift)/YSwing, 1);                   /* E-9 */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eg   = bwXfrFn(Epg, fn);                                   /* E-2. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
        break;
        case  8:
            t    = Y - (Cb-(1<<(CDepth-1)));                           /* E-25 */
            G    = fclampf(0, Y+(Cb-(1<<(CDepth-1))), ((1<<CDepth)-1));/* E-26 */
            B    = fclampf(0, t-(Cr-(1<<(CDepth-1))), ((1<<CDepth)-1));/* E-27 */
            R    = fclampf(0, t+(Cr-(1<<(CDepth-1))), ((1<<CDepth)-1));/* E-28 */
            Epr  = fclampf(0, (R-YShift)/YSwing, 1);                   /* E-7 */
            Epg  = fclampf(0, (G-YShift)/YSwing, 1);                   /* E-8 */
            Epb  = fclampf(0, (B-YShift)/YSwing, 1);                   /* E-9 */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eg   = bwXfrFn(Epg, fn);                                   /* E-2. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
        break;
    }
    
    return make_float3(Eg, Eb, Er);
}




/**
 * @brief Conversion Kernels
 */

/**
 * @brief Conversion from gbrp to yuv420p
 * 
 * Every thread handles a 2x2 pixel block, due to yuv420p's inherent chroma
 * subsampling.
 */

template<int  DST_CM         = AVCOL_SPC_BT470BG,
         int  SRC_CM         = AVCOL_SPC_BT470BG,
         int  FN             = AVCOL_TRC_IEC61966_2_1,
         bool SRC_FULL_SWING = true,
         bool DST_FULL_SWING = true,
         bool LERP           = true,
         unsigned SRC_YDEPTH = 8,
         unsigned SRC_CDEPTH = 8,
         unsigned DST_YDEPTH = 8,
         unsigned DST_CDEPTH = 8>
static __global__ void KERNEL_BOUNDS
i2vKernel_gbrp_to_yuv420p(void* __restrict const dstYPtr,
                          void* __restrict const dstCbPtr,
                          void* __restrict const dstCrPtr,
                          const unsigned         dstH,
                          const unsigned         dstW,
                          const unsigned         embedX, const unsigned embedY, const unsigned embedW, const unsigned embedH,
                          const unsigned         cropX,  const unsigned cropY,  const unsigned cropW,  const unsigned cropH,
                          const float            kx,     const float    ky,
                          const float            subX0,  const float    subY0,
                          const float            subX1,  const float    subY1,
                          const float            subX2,  const float    subY2,
                          const float            offX0,  const float    offY0,
                          const float            offX1,  const float    offY1,
                          const float            offX2,  const float    offY2,
                          cudaTextureObject_t    srcGPlane,
                          cudaTextureObject_t    srcBPlane,
                          cudaTextureObject_t    srcRPlane){
    /* Compute Destination Coordinates */
    const unsigned x = 2*(blockDim.x*blockIdx.x + threadIdx.x);
    const unsigned y = 2*(blockDim.y*blockIdx.y + threadIdx.y);
    if(x >= dstW || y >= dstH){return;}
    
    /* Compute Destination Y,U,V Pointers */
    uchar1* const __restrict__ dstPtr0 = (uchar1*)dstYPtr  + (y/1)*dstW/1 + (x/1);
    uchar1* const __restrict__ dstPtr1 = (uchar1*)dstCbPtr + (y/2)*dstW/2 + (x/2);
    uchar1* const __restrict__ dstPtr2 = (uchar1*)dstCrPtr + (y/2)*dstW/2 + (x/2);
    
    /**
     * Compute Master Quad Source Coordinates
     * 
     * The points (X0,Y0), (X1,Y1), (X2,Y2), (X3,Y3) represent the exact
     * coordinates at which we must sample for the Y samples 0,1,2,3 respectively.
     * The Cb/Cr sample corresponding to this quad is interpolated from the
     * samples drawn above.
     * 
     * The destination image is padded and the source image is cropped. This
     * means that when starting from the raw destination coordinates, we must:
     * 
     *    1. Subtract (embedX, embedY), since those are the destination embedding
     *       offsets in the target buffer.
     *    2. Clamp to (embedW-1, embedH-1), since those are the true destination
     *       bounds into the target buffer and any pixel beyond that limit
     *       should replicate the value found at the edge.
     *    3. Scale by the factors (kx, ky).
     *    4. Clamp to (cropW-1, cropH-1), since those are the bounds of the
     *       cropped source image.
     *    5. Add (cropX, cropY), since those are the source offsets into the
     *       cropped source buffer that should correspond to (embedX, embedY)
     *       in the destination buffer.
     */
    
    /* Plane 0: Y */
    const float X0p0=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX0+offX0;
    const float X1p0=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX0+offX0;
    const float X2p0=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX0+offX0;
    const float X3p0=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX0+offX0;
    const float Y0p0=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY0+offY0;
    const float Y1p0=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY0+offY0;
    const float Y2p0=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY0+offY0;
    const float Y3p0=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY0+offY0;
    /* Plane 1: Cb */
    const float X0p1=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX1+offX1;
    const float X1p1=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX1+offX1;
    const float X2p1=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX1+offX1;
    const float X3p1=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX1+offX1;
    const float Y0p1=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY1+offY1;
    const float Y1p1=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY1+offY1;
    const float Y2p1=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY1+offY1;
    const float Y3p1=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY1+offY1;
    /* Plane 2: Cr */
    const float X0p2=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX2+offX2;
    const float X1p2=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX2+offX2;
    const float X2p2=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX2+offX2;
    const float X3p2=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX2+offX2;
    const float Y0p2=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY2+offY2;
    const float Y1p2=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY2+offY2;
    const float Y2p2=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY2+offY2;
    const float Y3p2=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY2+offY2;
    
    
    /* Compute and output pixel depending on interpolation policy. */
    if(LERP){
        /**
         * We will perform linear interpolation manually between sample points.
         * For this reason we need the coordinates and values of 4 samples for
         * each of the 4 YUV pixels this thread is computing. In ASCII art:
         * 
         *  (0,0)
         *      +-----------> X
         *      |           Xb
         *      |
         *      |
         *      v Yb        0  1
         *      Y             P
         *                  2  3
         * 
         * We must consider three planes separately:
         * 
         *   - Plane 0 is luma                   or G
         *   - Plane 1 is chroma blue-difference or B
         *   - Plane 2 is chroma red -difference or R
         * 
         * If sample i's plane j sample is at (Xipj, Yipj), the quad used to
         * manually interpolate it has coordinates
         * 
         *   - (Xipjq0, Yipjq0)
         *   - (Xipjq1, Yipjq1)
         *   - (Xipjq2, Yipjq2)
         *   - (Xipjq3, Yipjq3)
         * 
         * And the interpolation within the quad is done using the scalar pair
         * 
         *   - (Xipjf, Yipjf)
         * 
         * or for an entire pixel, with the vector pair
         * 
         *   - (Xipf, Yipf)
         */
        
        const float X0p0b=floorf(X0p0), X0p0f=X0p0-X0p0b;
        const float X1p0b=floorf(X1p0), X1p0f=X1p0-X1p0b;
        const float X2p0b=floorf(X2p0), X2p0f=X2p0-X2p0b;
        const float X3p0b=floorf(X3p0), X3p0f=X3p0-X3p0b;
        const float X0p1b=floorf(X0p1), X0p1f=X0p1-X0p1b;
        const float X1p1b=floorf(X1p1), X1p1f=X1p1-X1p1b;
        const float X2p1b=floorf(X2p1), X2p1f=X2p1-X2p1b;
        const float X3p1b=floorf(X3p1), X3p1f=X3p1-X3p1b;
        const float X0p2b=floorf(X0p2), X0p2f=X0p2-X0p2b;
        const float X1p2b=floorf(X1p2), X1p2f=X1p2-X1p2b;
        const float X2p2b=floorf(X2p2), X2p2f=X2p2-X2p2b;
        const float X3p2b=floorf(X3p2), X3p2f=X3p2-X3p2b;
        
        const float Y0p0b=floorf(Y0p0), Y0p0f=Y0p0-Y0p0b;
        const float Y1p0b=floorf(Y1p0), Y1p0f=Y1p0-Y1p0b;
        const float Y2p0b=floorf(Y2p0), Y2p0f=Y2p0-Y2p0b;
        const float Y3p0b=floorf(Y3p0), Y3p0f=Y3p0-Y3p0b;
        const float Y0p1b=floorf(Y0p1), Y0p1f=Y0p1-Y0p1b;
        const float Y1p1b=floorf(Y1p1), Y1p1f=Y1p1-Y1p1b;
        const float Y2p1b=floorf(Y2p1), Y2p1f=Y2p1-Y2p1b;
        const float Y3p1b=floorf(Y3p1), Y3p1f=Y3p1-Y3p1b;
        const float Y0p2b=floorf(Y0p2), Y0p2f=Y0p2-Y0p2b;
        const float Y1p2b=floorf(Y1p2), Y1p2f=Y1p2-Y1p2b;
        const float Y2p2b=floorf(Y2p2), Y2p2f=Y2p2-Y2p2b;
        const float Y3p2b=floorf(Y3p2), Y3p2f=Y3p2-Y3p2b;
        
        
        /* Sample from texture at the computed coordinates. */
        const float3 srcPix00 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+0, Y0p0b+0), tex2D<unsigned char>(srcBPlane, X0p1b+0, Y0p1b+0), tex2D<unsigned char>(srcRPlane, X0p2b+0, Y0p2b+0));
        const float3 srcPix01 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+1, Y0p0b+0), tex2D<unsigned char>(srcBPlane, X0p1b+1, Y0p1b+0), tex2D<unsigned char>(srcRPlane, X0p2b+1, Y0p2b+0));
        const float3 srcPix02 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+0, Y0p0b+1), tex2D<unsigned char>(srcBPlane, X0p1b+0, Y0p1b+1), tex2D<unsigned char>(srcRPlane, X0p2b+0, Y0p2b+1));
        const float3 srcPix03 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+1, Y0p0b+1), tex2D<unsigned char>(srcBPlane, X0p1b+1, Y0p1b+1), tex2D<unsigned char>(srcRPlane, X0p2b+1, Y0p2b+1));
        const float3 srcPix10 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+0, Y1p0b+0), tex2D<unsigned char>(srcBPlane, X1p1b+0, Y1p1b+0), tex2D<unsigned char>(srcRPlane, X1p2b+0, Y1p2b+0));
        const float3 srcPix11 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+1, Y1p0b+0), tex2D<unsigned char>(srcBPlane, X1p1b+1, Y1p1b+0), tex2D<unsigned char>(srcRPlane, X1p2b+1, Y1p2b+0));
        const float3 srcPix12 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+0, Y1p0b+1), tex2D<unsigned char>(srcBPlane, X1p1b+0, Y1p1b+1), tex2D<unsigned char>(srcRPlane, X1p2b+0, Y1p2b+1));
        const float3 srcPix13 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+1, Y1p0b+1), tex2D<unsigned char>(srcBPlane, X1p1b+1, Y1p1b+1), tex2D<unsigned char>(srcRPlane, X1p2b+1, Y1p2b+1));
        const float3 srcPix20 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+0, Y2p0b+0), tex2D<unsigned char>(srcBPlane, X2p1b+0, Y2p1b+0), tex2D<unsigned char>(srcRPlane, X2p2b+0, Y2p2b+0));
        const float3 srcPix21 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+1, Y2p0b+0), tex2D<unsigned char>(srcBPlane, X2p1b+1, Y2p1b+0), tex2D<unsigned char>(srcRPlane, X2p2b+1, Y2p2b+0));
        const float3 srcPix22 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+0, Y2p0b+1), tex2D<unsigned char>(srcBPlane, X2p1b+0, Y2p1b+1), tex2D<unsigned char>(srcRPlane, X2p2b+0, Y2p2b+1));
        const float3 srcPix23 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+1, Y2p0b+1), tex2D<unsigned char>(srcBPlane, X2p1b+1, Y2p1b+1), tex2D<unsigned char>(srcRPlane, X2p2b+1, Y2p2b+1));
        const float3 srcPix30 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+0, Y3p0b+0), tex2D<unsigned char>(srcBPlane, X3p1b+0, Y3p1b+0), tex2D<unsigned char>(srcRPlane, X3p2b+0, Y3p2b+0));
        const float3 srcPix31 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+1, Y3p0b+0), tex2D<unsigned char>(srcBPlane, X3p1b+1, Y3p1b+0), tex2D<unsigned char>(srcRPlane, X3p2b+1, Y3p2b+0));
        const float3 srcPix32 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+0, Y3p0b+1), tex2D<unsigned char>(srcBPlane, X3p1b+0, Y3p1b+1), tex2D<unsigned char>(srcRPlane, X3p2b+0, Y3p2b+1));
        const float3 srcPix33 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+1, Y3p0b+1), tex2D<unsigned char>(srcBPlane, X3p1b+1, Y3p1b+1), tex2D<unsigned char>(srcRPlane, X3p2b+1, Y3p2b+1));
        
        
        /* Backward Colorspace and Transfer (EOTF). */
        const float3 linPix00 = bwXfrCol(srcPix00, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix01 = bwXfrCol(srcPix01, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix02 = bwXfrCol(srcPix02, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix03 = bwXfrCol(srcPix03, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix10 = bwXfrCol(srcPix10, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix11 = bwXfrCol(srcPix11, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix12 = bwXfrCol(srcPix12, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix13 = bwXfrCol(srcPix13, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix20 = bwXfrCol(srcPix20, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix21 = bwXfrCol(srcPix21, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix22 = bwXfrCol(srcPix22, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix23 = bwXfrCol(srcPix23, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix30 = bwXfrCol(srcPix30, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix31 = bwXfrCol(srcPix31, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix32 = bwXfrCol(srcPix32, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix33 = bwXfrCol(srcPix33, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        
        
        /* Linear Interpolation */
        const float3 X0pf     = make_float3(X0p0f, X0p1f, X0p2f);
        const float3 X1pf     = make_float3(X1p0f, X1p1f, X1p2f);
        const float3 X2pf     = make_float3(X2p0f, X2p1f, X2p2f);
        const float3 X3pf     = make_float3(X3p0f, X3p1f, X3p2f);
        const float3 Y0pf     = make_float3(Y0p0f, Y0p1f, Y0p2f);
        const float3 Y1pf     = make_float3(Y1p0f, Y1p1f, Y1p2f);
        const float3 Y2pf     = make_float3(Y2p0f, Y2p1f, Y2p2f);
        const float3 Y3pf     = make_float3(Y3p0f, Y3p1f, Y3p2f);
        const float3 linPixD0 = lerpv3(lerpv3(linPix00, linPix01, X0pf), lerpv3(linPix02, linPix03, X0pf), Y0pf);
        const float3 linPixD1 = lerpv3(lerpv3(linPix10, linPix11, X1pf), lerpv3(linPix12, linPix13, X1pf), Y1pf);
        const float3 linPixD2 = lerpv3(lerpv3(linPix20, linPix21, X2pf), lerpv3(linPix22, linPix23, X2pf), Y2pf);
        const float3 linPixD3 = lerpv3(lerpv3(linPix30, linPix31, X3pf), lerpv3(linPix32, linPix33, X3pf), Y3pf);
        const float3 linPixD4 = lerps3(lerps3(linPixD0, linPixD1, 0.5f), lerps3(linPixD2, linPixD3, 0.5f), 0.5f);
        
        
        /* Forward Colorspace and Transfer (OETF). */
        const float3 dstPixD0 = fwXfrCol(linPixD0, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPixD1 = fwXfrCol(linPixD1, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPixD2 = fwXfrCol(linPixD2, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPixD3 = fwXfrCol(linPixD3, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPixD4 = fwXfrCol(linPixD4, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        
        
        /* Store YUV. */
        dstPtr0[0*dstW/1/1+0]   = make_uchar1(roundf(fclampf(0.f, dstPixD0.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[0*dstW/1/1+1]   = make_uchar1(roundf(fclampf(0.f, dstPixD1.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[1*dstW/1/1+0]   = make_uchar1(roundf(fclampf(0.f, dstPixD2.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[1*dstW/1/1+1]   = make_uchar1(roundf(fclampf(0.f, dstPixD3.x, (1<<DST_YDEPTH)-1)));
        dstPtr1[0*dstW/2/2+0]   = make_uchar1(roundf(fclampf(0.f, dstPixD4.y, (1<<DST_CDEPTH)-1)));
        dstPtr2[0*dstW/2/2+0]   = make_uchar1(roundf(fclampf(0.f, dstPixD4.z, (1<<DST_CDEPTH)-1)));
    }else{
        /**
         * We perform nearest-sample point fetch for 4 Y, 1 Cb and 1 Cr samples.
         * 
         * Since YUV420 is 2x2 chroma-subsampled, we select the top left
         * pixel's chroma information, given that we're forbidden to interpolate.
         * 
         * We must consider three planes separately:
         * 
         *   - Plane 0 is luma                   or G
         *   - Plane 1 is chroma blue-difference or B
         *   - Plane 2 is chroma red -difference or R
         */
        
        const float X0p0b=roundf(X0p0);
        const float X1p0b=roundf(X1p0);
        const float X2p0b=roundf(X2p0);
        const float X3p0b=roundf(X3p0);
        const float X0p1b=roundf(X0p1);
        const float X1p1b=roundf(X1p1);
        const float X2p1b=roundf(X2p1);
        const float X3p1b=roundf(X3p1);
        const float X0p2b=roundf(X0p2);
        const float X1p2b=roundf(X1p2);
        const float X2p2b=roundf(X2p2);
        const float X3p2b=roundf(X3p2);
        
        const float Y0p0b=roundf(Y0p0);
        const float Y1p0b=roundf(Y1p0);
        const float Y2p0b=roundf(Y2p0);
        const float Y3p0b=roundf(Y3p0);
        const float Y0p1b=roundf(Y0p1);
        const float Y1p1b=roundf(Y1p1);
        const float Y2p1b=roundf(Y2p1);
        const float Y3p1b=roundf(Y3p1);
        const float Y0p2b=roundf(Y0p2);
        const float Y1p2b=roundf(Y1p2);
        const float Y2p2b=roundf(Y2p2);
        const float Y3p2b=roundf(Y3p2);
        
        
        /* Sample from texture at the computed coordinates. */
        const float3 srcPix0 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+0, Y0p0b+0), tex2D<unsigned char>(srcBPlane, X0p1b+0, Y0p1b+0), tex2D<unsigned char>(srcRPlane, X0p2b+0, Y0p2b+0));
        const float3 srcPix1 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+1, Y1p0b+0), tex2D<unsigned char>(srcBPlane, X1p1b+1, Y1p1b+0), tex2D<unsigned char>(srcRPlane, X1p2b+1, Y1p2b+0));
        const float3 srcPix2 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+0, Y2p0b+1), tex2D<unsigned char>(srcBPlane, X2p1b+0, Y2p1b+1), tex2D<unsigned char>(srcRPlane, X2p2b+0, Y2p2b+1));
        const float3 srcPix3 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+1, Y3p0b+1), tex2D<unsigned char>(srcBPlane, X3p1b+1, Y3p1b+1), tex2D<unsigned char>(srcRPlane, X3p2b+1, Y3p2b+1));
        
        
        /* Backward Colorspace and Transfer (EOTF). */
        const float3 linPix0 = bwXfrCol(srcPix0, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix1 = bwXfrCol(srcPix1, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix2 = bwXfrCol(srcPix2, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix3 = bwXfrCol(srcPix3, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        
        
        /* Forward Colorspace and Transfer (OETF). */
        const float3 dstPix0 = fwXfrCol(linPix0, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPix1 = fwXfrCol(linPix1, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPix2 = fwXfrCol(linPix2, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPix3 = fwXfrCol(linPix3, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        
        
        /* Store YUV. */
        dstPtr0[0*dstW/1/1+0]   = make_uchar1(roundf(fclampf(0.f, dstPix0.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[0*dstW/1/1+1]   = make_uchar1(roundf(fclampf(0.f, dstPix1.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[1*dstW/1/1+0]   = make_uchar1(roundf(fclampf(0.f, dstPix2.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[1*dstW/1/1+1]   = make_uchar1(roundf(fclampf(0.f, dstPix3.x, (1<<DST_YDEPTH)-1)));
        dstPtr1[0*dstW/2/2+0]   = make_uchar1(roundf(fclampf(0.f, dstPix0.y, (1<<DST_CDEPTH)-1)));
        dstPtr2[0*dstW/2/2+0]   = make_uchar1(roundf(fclampf(0.f, dstPix0.z, (1<<DST_CDEPTH)-1)));
    }
}



/**
 * @brief Do manual filtering.
 * 
 * @return 0 if successful; !0 otherwise.
 */

extern int  i2v_cuda_filter(UNIVERSE* u, AVFrame* dst, AVFrame* src, BENZINA_GEOM* geom){
    struct cudaResourceDesc resDesc0, resDesc1, resDesc2;
    struct cudaTextureDesc  texDesc0, texDesc1, texDesc2;
    cudaTextureObject_t     texObj0,  texObj1,  texObj2;
    dim3                    Dg, Db;
    cudaError_t             err         = cudaSuccess;
    void*                   cudaIPlane0 = NULL;
    void*                   cudaIPlane1 = NULL;
    void*                   cudaIPlane2 = NULL;
    void*                   cudaOPlane0 = NULL;
    void*                   cudaOPlane1 = NULL;
    void*                   cudaOPlane2 = NULL;
    
#if 0
    printf("%s\n", av_get_pix_fmt_name((AVPixelFormat)src->format));
    printf("%d\n", av_pix_fmt_count_planes((AVPixelFormat)src->format));
    printf("%dx%d\n", src->width, src->height);
    printf("%p\n", src->data);
    printf("\t%p  +  %d\n", src->data[0], src->linesize[0]);
    printf("\t%p  +  %d\n", src->data[1], src->linesize[1]);
    printf("\t%p  +  %d\n", src->data[2], src->linesize[2]);
    printf("%s\n", av_get_pix_fmt_name((AVPixelFormat)dst->format));
    printf("%d\n", av_pix_fmt_count_planes((AVPixelFormat)dst->format));
    printf("%dx%d\n", dst->width, dst->height);
    printf("%p\n", dst->data);
    printf("\t%p  +  %d\n", dst->data[0], dst->linesize[0]);
    printf("\t%p  +  %d\n", dst->data[1], dst->linesize[1]);
    printf("\t%p  +  %d\n", dst->data[2], dst->linesize[2]);
#endif
    memset(&resDesc0, 0, sizeof(resDesc0));
    memset(&resDesc1, 0, sizeof(resDesc1));
    memset(&resDesc2, 0, sizeof(resDesc2));
    memset(&texDesc0, 0, sizeof(texDesc0));
    memset(&texDesc1, 0, sizeof(texDesc1));
    memset(&texDesc2, 0, sizeof(texDesc2));
    texObj0 = 0;
    texObj1 = 0;
    texObj2 = 0;
    err = cudaSetDevice(0);
    err = cudaMalloc(&cudaIPlane0, src->width *src->height /1/1);
    err = cudaMalloc(&cudaIPlane1, src->width *src->height /1/1);
    err = cudaMalloc(&cudaIPlane2, src->width *src->height /1/1);
    err = cudaMalloc(&cudaOPlane0, dst->width*dst->height/1/1);
    err = cudaMalloc(&cudaOPlane1, dst->width*dst->height/2/2);
    err = cudaMalloc(&cudaOPlane2, dst->width*dst->height/2/2);
    err = cudaMemset(cudaOPlane0,   0, dst->width*dst->height/1/1);
    err = cudaMemset(cudaOPlane1, 128, dst->width*dst->height/2/2);
    err = cudaMemset(cudaOPlane2, 255, dst->width*dst->height/2/2);
    err = cudaMemcpy2D(cudaIPlane0, src->width, src->data[0], src->linesize[0], src->width, src->height, cudaMemcpyHostToDevice);
    err = cudaMemcpy2D(cudaIPlane1, src->width, src->data[1], src->linesize[1], src->width, src->height, cudaMemcpyHostToDevice);
    err = cudaMemcpy2D(cudaIPlane2, src->width, src->data[2], src->linesize[2], src->width, src->height, cudaMemcpyHostToDevice);
    
    resDesc0.resType                  = cudaResourceTypePitch2D;
    resDesc0.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
    resDesc0.res.pitch2D.desc.x       = 8;
    resDesc0.res.pitch2D.desc.y       = 0;
    resDesc0.res.pitch2D.desc.z       = 0;
    resDesc0.res.pitch2D.desc.w       = 0;
    resDesc0.res.pitch2D.devPtr       = cudaIPlane0;
    resDesc0.res.pitch2D.height       = src->height;
    resDesc0.res.pitch2D.width        = src->width;
    resDesc0.res.pitch2D.pitchInBytes = src->linesize[0];
    texDesc0.addressMode[0]           = cudaAddressModeClamp;
    texDesc0.addressMode[1]           = cudaAddressModeClamp;
    texDesc0.filterMode               = cudaFilterModePoint;
    texDesc0.readMode                 = cudaReadModeElementType;
    texDesc0.sRGB                     = 0;
    texDesc0.normalizedCoords         = 0;
    
    resDesc1.resType                  = cudaResourceTypePitch2D;
    resDesc1.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
    resDesc1.res.pitch2D.desc.x       = 8;
    resDesc1.res.pitch2D.desc.y       = 0;
    resDesc1.res.pitch2D.desc.z       = 0;
    resDesc1.res.pitch2D.desc.w       = 0;
    resDesc1.res.pitch2D.devPtr       = cudaIPlane1;
    resDesc1.res.pitch2D.height       = src->height;
    resDesc1.res.pitch2D.width        = src->width;
    resDesc1.res.pitch2D.pitchInBytes = src->linesize[1];
    texDesc1.addressMode[0]           = cudaAddressModeClamp;
    texDesc1.addressMode[1]           = cudaAddressModeClamp;
    texDesc1.filterMode               = cudaFilterModePoint;
    texDesc1.readMode                 = cudaReadModeElementType;
    texDesc1.sRGB                     = 0;
    texDesc1.normalizedCoords         = 0;
    
    resDesc2.resType                  = cudaResourceTypePitch2D;
    resDesc2.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
    resDesc2.res.pitch2D.desc.x       = 8;
    resDesc2.res.pitch2D.desc.y       = 0;
    resDesc2.res.pitch2D.desc.z       = 0;
    resDesc2.res.pitch2D.desc.w       = 0;
    resDesc2.res.pitch2D.devPtr       = cudaIPlane2;
    resDesc2.res.pitch2D.height       = src->height;
    resDesc2.res.pitch2D.width        = src->width;
    resDesc2.res.pitch2D.pitchInBytes = src->linesize[2];
    texDesc2.addressMode[0]           = cudaAddressModeClamp;
    texDesc2.addressMode[1]           = cudaAddressModeClamp;
    texDesc2.filterMode               = cudaFilterModePoint;
    texDesc2.readMode                 = cudaReadModeElementType;
    texDesc2.sRGB                     = 0;
    texDesc2.normalizedCoords         = 0;
    
    err = cudaCreateTextureObject(&texObj0, &resDesc0, &texDesc0, NULL);
    err = cudaCreateTextureObject(&texObj1, &resDesc1, &texDesc1, NULL);
    err = cudaCreateTextureObject(&texObj2, &resDesc2, &texDesc2, NULL);
    
    int dstW  = dst->width,
        dstH  = dst->height;
    Db.x = 32;
    Db.y = 32;
    Db.z =  1;
    Dg.x = (dstW/2+Db.x-1)/Db.x;
    Dg.y = (dstH/2+Db.y-1)/Db.y;
    Dg.z =  1;
    i2vKernel_gbrp_to_yuv420p
    <AVCOL_SPC_BT470BG, AVCOL_SPC_RGB, AVCOL_TRC_IEC61966_2_1, true, true, true>
    <<<Dg, Db>>>(
        cudaOPlane0,
        cudaOPlane1,
        cudaOPlane2,
        dstH,
        dstW,
        rect2d_x(&geom->o.canvas), rect2d_y(&geom->o.canvas), rect2d_w(&geom->o.canvas), rect2d_h(&geom->o.canvas),
        rect2d_x(&geom->o.source), rect2d_y(&geom->o.source), rect2d_w(&geom->o.source), rect2d_h(&geom->o.source),
        1.0f,            1.0f,
        1.0f,            1.0f,
        1.0f,            1.0f,
        1.0f,            1.0f,
        0,0,
        0,0,
        0,0,
        texObj0,
        texObj1,
        texObj2
    );
    
    err = cudaMemcpy2D(dst->data[0], dst->linesize[0], cudaOPlane0, dst->width/1, dst->width/1, dst->height/1, cudaMemcpyDeviceToHost);
    err = cudaMemcpy2D(dst->data[1], dst->linesize[1], cudaOPlane1, dst->width/2, dst->width/2, dst->height/2, cudaMemcpyDeviceToHost);
    err = cudaMemcpy2D(dst->data[2], dst->linesize[2], cudaOPlane2, dst->width/2, dst->width/2, dst->height/2, cudaMemcpyDeviceToHost);
    
    cudaFree(cudaIPlane0);
    cudaFree(cudaIPlane1);
    cudaFree(cudaIPlane2);
    cudaFree(cudaOPlane0);
    cudaFree(cudaOPlane1);
    cudaFree(cudaOPlane2);
    
    printf("%d\n", err);
    
    return 0;
}


/**
 * Other References:
 * 
 *   - The "Continuum Magic Kernel Sharp" method:
 * 
 *                { 0.0             ,         x <= -1.5
 *                { 0.5(x+1.5)**2   , -1.5 <= x <= -0.5
 *         m(x) = { 0.75 - (x)**2   , -0.5 <= x <= +0.5
 *                { 0.5(x-1.5)**2   , +0.5 <= x <= +1.5
 *                { 0.0             , +1.5 <= x
 *     
 *     followed by the sharpening post-filter [-0.25, +1.5, -0.25].
 *     http://www.johncostella.com/magic/
 */

