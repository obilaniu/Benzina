/* Include Guard */
#ifndef INCLUDE_BENZINA_ITU_H26X_BITSTREAM_H
#define INCLUDE_BENZINA_ITU_H26X_BITSTREAM_H


/**
 * Includes
 */

#include <stdint.h>
#include <string.h>
#include "benzina/bits.h"
#include "benzina/bitops.h"
#include "benzina/endian.h"
#include "benzina/inline.h"
#include "benzina/visibility.h"


/* Defines */


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Data Structures Forward Declarations and Typedefs */
typedef struct BENZ_H26XBS             BENZ_H26XBS;



/* Data Structure Definitions */

/**
 * @brief ITU H.26x-standard bitstream parser.
 */

struct BENZ_H26XBS{
    uint8_t        rbsp[256-3*sizeof(uint64_t)
                           -3*sizeof(uint8_t*)];
    uint64_t       sreg;
    uint64_t       sregrem;
    uint8_t*       rbsphead;
    uint8_t*       rbsptail;
    const uint8_t* naluptr;
    uint64_t       nalulen;
};


/* Public Function Forward Declarations */
BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h26xbs_fill(BENZ_H26XBS* bs);

BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h26xbs_read_1b(BENZ_H26XBS* bs);
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h26xbs_read_un(BENZ_H26XBS* bs, int n);
BENZINA_PUBLIC BENZINA_INLINE int64_t     benz_itu_h26xbs_read_sn(BENZ_H26XBS* bs, int n);
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h26xbs_read_ue(BENZ_H26XBS* bs);
BENZINA_PUBLIC BENZINA_INLINE int64_t     benz_itu_h26xbs_read_se(BENZ_H26XBS* bs);

BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_nb(BENZ_H26XBS* bs, int n);
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_ue(BENZ_H26XBS* bs);
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_se(BENZ_H26XBS* bs);



/* Public Function Declarations & Definitions */

/**
 * @brief Inline Function Definitions for ITU H.26x bitstream parsing support.
 * 
 * Both ITU-T H.264 and H.265 have the concept of a NAL unit containing an RBSP
 * (Raw Byte Sequence Payload), with escape emulation prevention bytes (EPBs).
 * These EPBs must be stripped to obtain the RBSP.
 */

/**
 * @brief Initialize Bitstream Parser.
 * 
 * @param [in]  bs           Bitstream Parser
 * @param [in]  nalu         Pointer to NALU to parse.
 * @param [in]  namubytelen  Length in bytes of NALU to parse.
 * @return 0
 */

BENZINA_PUBLIC                int         benz_itu_h26xbs_init(BENZ_H26XBS* bs,
                                                               const void*  nalu,
                                                               size_t       nalubytelen);

/**
 * @brief Fill Shift Register
 * @param [in]  bs  Bitstream Parser
 * @return 0
 */

BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h26xbs_fill(BENZ_H26XBS* bs){
    return 0;
}

/**
 * @brief Fill RBSP Buffer
 * @param [in]  bs  Bitstream Parser
 * @return 0
 */

BENZINA_PUBLIC                int         benz_itu_h26xbs_bigfill(BENZ_H26XBS* bs);

/**
 * @brief Read one bit, n unsigned or signed bits, an unsigned or signed
 *        Exp-Golomb value.
 * 
 * An unsigned Exp-Golomb number ue(x) is encoded as follows:
 *   1. Write floor(log2(x+1)) leading zeroes.
 *   2. Write x+1 in binary with no leading zeroes.
 * A signed Exp-Golomb number se(x) is encoded as follows:
 *   - ue(2x-1) if x>0
 *   - ue(-2x)  if x<=0
 * 
 * @param [in]  bs  Bitstream Parser
 * @param [in]  n   Number of bits to be read. At least 1 and at most 64.
 */

BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h26xbs_read_1b(BENZ_H26XBS* bs){
    int ret = (int64_t)bs->sreg < 0;
    bs->sreg <<= 1;
    bs->sregrem--;
    return ret;
}
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h26xbs_read_un(BENZ_H26XBS* bs, int n){
    uint64_t ret = bs->sreg >> (64-n);
    bs->sreg <<= n;
    bs->sregrem -= n;
    return ret;
}
BENZINA_PUBLIC BENZINA_INLINE int64_t     benz_itu_h26xbs_read_sn(BENZ_H26XBS* bs, int n){
    int64_t ret = (int64_t)bs->sreg >> (64-n);
    bs->sreg <<= n;
    bs->sregrem -= n;
    return ret;
}
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h26xbs_read_ue(BENZ_H26XBS* bs){
    uint64_t v;
    int r = benz_clz64(bs->sreg);
    bs->sreg <<= r;
    v = bs->sreg >> (63-r);
    bs->sreg <<= r+1;
    bs->sregrem -= 2*r+1;
    return v-1;
}
BENZINA_PUBLIC BENZINA_INLINE int64_t     benz_itu_h26xbs_read_se(BENZ_H26XBS* bs){
    uint64_t v = benz_itu_h26xbs_read_ue(bs);
    v = v&1 ? v+1 : -v;
    return (int64_t)v>>1;
}

/**
 * @brief Skip n bits, an unsigned or signed Exp-Golomb value.
 * 
 * @param [in]  bs  Bitstream Parser
 * @param [in]  n   Number of bits to be skipped.
 */

BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_nb(BENZ_H26XBS* bs, int n){
    bs->sreg <<= n;
    bs->sregrem -= n;
}
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_ue(BENZ_H26XBS* bs){
    benz_itu_h26xbs_skip_nb(bs, 2*benz_clz64(bs->sreg)+1);
}
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_se(BENZ_H26XBS* bs){
    benz_itu_h26xbs_skip_ue(bs);
}


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

