/* Include Guard */
#ifndef INCLUDE_BENZINA_ITU_H265_H
#define INCLUDE_BENZINA_ITU_H265_H


/**
 * Includes
 */

#include <stdint.h>
#include <string.h>
#include "benzina/bits.h"
#include "benzina/endian.h"
#include "benzina/inline.h"
#include "benzina/intops.h"
#include "benzina/visibility.h"


/* Defines */


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Data Structures Forward Declarations and Typedefs */
typedef void                           BENZ_ITU_H265_NALU;



/**
 * Function Declarations and Definitions for ITU-T H.265.
 */

BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_check_header(const BENZ_ITU_H265_NALU* p){
    int16_t v = benz_getbe16(p, 0);
    return  (v>=0) && (v&7);/* Ensure forbidden_zero_bit==0 && nuh_temporal_id_plus1>0 */
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_get_type(const BENZ_ITU_H265_NALU* p){
    return benz_getbe8(p, 0) >> 1;
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_vcl(const BENZ_ITU_H265_NALU* p){
    return benz_getbe8(p, 0) < 0x40;
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_idr(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_IDR_W_RADL << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_CRA_NUT    << 1);
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_idr_n_lp(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_IDR_N_LP << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_CRA_NUT  << 1);
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_vps(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_VPS_NUT << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_SPS_NUT << 1);
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_sps(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_SPS_NUT << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_PPS_NUT << 1);
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_pps(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_PPS_NUT << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_AUD_NUT << 1);
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_aud(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_AUD_NUT << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_EOS_NUT << 1);
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_eos(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_EOS_NUT << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_EOB_NUT << 1);
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_eob(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_EOB_NUT << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_FD_NUT  << 1);
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_fd(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_FD_NUT         << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_PREFIX_SEI_NUT << 1);
}
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h265_nalu_is_sei(const BENZ_ITU_H265_NALU* p){
    uint8_t b = benz_getbe8(p, 0);
    return b >= (BENZINA_H265_NALU_TYPE_PREFIX_SEI_NUT << 1) &&
           b <  (BENZINA_H265_NALU_TYPE_RSV_NVCL41     << 1);
}
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h265_nalu_slice_get_pps_id(const BENZ_ITU_H265_NALU* p){
    /* Impossible to have EPBs this early into valid slice */
    uint32_t v = benz_getbe32(p, 0)|1;
    uint32_t t = v >> 25;
    uint32_t e = (0x00FF0000UL >> t)&1;
    uint32_t s = 49+e;
    uint64_t w = (uint64_t)v << s;
    uint64_t d = benz_ueto64(w);
    return d;
}
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h265_nalu_pps_get_pps_id(const BENZ_ITU_H265_NALU* p){
    /* Impossible to have EPBs this early into valid PPS */
    uint32_t v = benz_getbe32(p, 2)|3;
    uint64_t w = (uint64_t)v << 32;
    uint64_t d = benz_ueto64(w);
    return d;
}
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h265_nalu_pps_get_sps_id(const BENZ_ITU_H265_NALU* p){
    /* Impossible to have EPBs this early into valid PPS */
    uint32_t v = benz_getbe32(p, 2)|3;
    uint64_t w = (uint64_t)v << 32;
    uint64_t c = 2*benz_clz64(w)+1;
    uint64_t d = benz_ueto64(w << c);
    return d;
}
BENZINA_PUBLIC                uint64_t    benz_itu_h265_nalu_sps_get_sps_id(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h265_nalu_sps_get_vps_id(const BENZ_ITU_H265_NALU* p){
    /* Impossible to have EPBs this early into valid SPS */
    return benz_getbe8(p, 2) >> 4;
}
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h265_nalu_vps_get_vps_id(const BENZ_ITU_H265_NALU* p){
    /* Impossible to have EPBs this early into valid VPS */
    return benz_getbe8(p, 2) >> 4;
}




/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

