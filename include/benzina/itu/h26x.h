/* Include Guard */
#ifndef INCLUDE_BENZINA_ITU_H26X_H
#define INCLUDE_BENZINA_ITU_H26X_H


/**
 * Includes
 */

#include <stdint.h>
#include <string.h>
#include "benzina/bits.h"
#include "benzina/intops.h"
#include "benzina/endian.h"
#include "benzina/inline.h"
#include "benzina/visibility.h"


/* Defines */
#define BENZ_H26XBS_ERR_CORRUPT       1  /* The bitstream is corrupt. */
#define BENZ_H26XBS_ERR_OVERREAD      2  /* An unfilled bit has been returned. */


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Data Structures Forward Declarations and Typedefs */
typedef struct BENZ_ITU_H26XBS         BENZ_ITU_H26XBS;



/* Data Structure Definitions */

/**
 * @brief ITU H.26x-standard bitstream parser.
 */

struct BENZ_ITU_H26XBS{
    /** @brief Shift register. */
    uint64_t       sreg;
    
    /** @brief Error condition mask */
    uint32_t       errmask;
    
    /**
     * @brief RBSP head, tail and shift register offset:
     * 
     * Tail offset:
     *   - Bit-offset from which bits were last read from the rbsp buffer into
     *     the shift register.
     *   - Multiple of 8 following a fill57() call, otherwise arbitrary.
     * 
     * Head offset:
     *   - Bit-offset at which bits are written into the rbsp buffer.
     *   - Multiple of 8.
     * 
     * Shift Register offset:
     *   - Bit-offset at which bits would be read from the rbsp buffer if a
     *     fill were to occur.
     */
    
    uint32_t       sregoff;
    uint32_t       tailoff;
    uint32_t       headoff;
    
    
    /** @brief Length of NALU in bytes. */
    uint64_t       nalulen;
    
    /** @brief Pointer to NALU bytes. */
    const uint8_t* naluptr;
    
    /**
     * @brief RBSP ring buffer.
     * 
     * Consists of a 192-byte main area plus a 64-byte "vectorization" area.
     * The vectorization area exists
     * 
     *   1) To allow vector instructions to write with an "excess" that is then
     *      "shifted" by copying it backwards by multiples of 64 bytes.
     *   2) To allow a small shift-register fill to overread by up to 64 bytes.
     */
    
    uint8_t        rbsp[128+64+64];
};


/* Public Function Forward Declarations */
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_fill57b(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_fill64b(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_fill8B (BENZ_ITU_H26XBS* bs);

BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h26xbs_read_1b(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h26xbs_read_un(BENZ_ITU_H26XBS* bs, unsigned n);
BENZINA_PUBLIC BENZINA_INLINE int64_t     benz_itu_h26xbs_read_sn(BENZ_ITU_H26XBS* bs, unsigned n);
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h26xbs_read_ue(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_INLINE int64_t     benz_itu_h26xbs_read_se(BENZ_ITU_H26XBS* bs);

BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_xn(BENZ_ITU_H26XBS* bs, unsigned n);
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_xe(BENZ_ITU_H26XBS* bs);

BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h26xbs_eos    (BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_INLINE uint32_t    benz_itu_h26xbs_err    (BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_INLINE uint32_t    benz_itu_h26xbs_markcor(BENZ_ITU_H26XBS* bs);

BENZINA_PUBLIC                void        benz_itu_h26xbs_bigfill(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC                void        benz_itu_h26xbs_bigskip(BENZ_ITU_H26XBS* bs, uint64_t n);



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
 * @param [in]  nalubytelen  Length in bytes of NALU to parse.
 * @return 0
 */

BENZINA_PUBLIC                void        benz_itu_h26xbs_init(BENZ_ITU_H26XBS* bs,
                                                               const void*  nalu,
                                                               size_t       nalubytelen);

/**
 * @brief Lightweight 57+-bit/64-bit/8-byte Fill of Shift Register.
 * 
 * Except at the end of the stream, this function requires that the RBSP buffer
 * have at least 8/9 bytes available, and guarantees in return that the shift
 * register will have at least 57/64 bits available.
 * 
 * The 8B variant additionally requires byte alignment.
 * 
 * Idempotent. Unsafe. Implemented in branch-free fashion.
 * 
 * @param [in]  bs  Bitstream Parser
 */

BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_fill57b(BENZ_ITU_H26XBS* bs){
    bs->tailoff = bs->sregoff & ~7;
    bs->sreg    = benz_getbe64(bs->rbsp, bs->sregoff>>3) << (bs->sregoff & 7);
}
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_fill64b(BENZ_ITU_H26XBS* bs){
    uint32_t sregoff = bs->sregoff;
    uint8_t* rbspptr = bs->rbsp + (sregoff >> 3);
    
    bs->tailoff = sregoff;
    bs->sreg    = benz_getbe64(rbspptr, 0) <<      (sregoff & 7) |
        (uint64_t)benz_getbe8 (rbspptr, 8) >> (8 - (sregoff & 7));
}
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_fill8B (BENZ_ITU_H26XBS* bs){
    bs->tailoff = bs->sregoff;
    bs->sreg    = benz_getbe64(bs->rbsp, bs->sregoff>>3);
}

/**
 * @brief Read one bit, n unsigned or signed bits, an unsigned or signed
 *        Exp-Golomb value from the shift-register.
 * 
 * An unsigned Exp-Golomb number ue(x) is encoded as follows:
 *   1. Write floor(log2(x+1)) leading zeroes.
 *   2. Write x+1 in binary with no leading zeroes.
 * A signed Exp-Golomb number se(x) is encoded as follows:
 *   - ue(2x-1) if x>0
 *   - ue(-2x)  if x<=0
 * To decode, therefore, one must invert this process:
 *   - Count leading zeros
 *   - Extract one more bits than there were leading zeroes.
 * 
 * @param [in]  bs  Bitstream Parser
 * @param [in]  n   Number of bits to be read. At least 1 and at most 64.
 * @return The bit or signed/unsigned value in question.
 */

BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h26xbs_read_1b(BENZ_ITU_H26XBS* bs){
    uint64_t v;
    int      ret;
    
    v            = bs->sreg;
    ret          = (int64_t)v < 0;
    bs->sreg     = v << 1;
    bs->sregoff += 1;
    return ret;
}
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h26xbs_read_un(BENZ_ITU_H26XBS* bs, unsigned n){
    uint64_t v, ret;
    
    v            = bs->sreg;
    ret          = v >> (64-n);
    bs->sreg     = v << n;
    bs->sregoff += n;
    return ret;
}
BENZINA_PUBLIC BENZINA_INLINE int64_t     benz_itu_h26xbs_read_sn(BENZ_ITU_H26XBS* bs, unsigned n){
    uint64_t v;
    int64_t  ret;
    
    v            = bs->sreg;
    ret          = (int64_t)v >> (64-n);
    bs->sreg     = v << n;
    bs->sregoff += n;
    return ret;
}
BENZINA_PUBLIC BENZINA_INLINE uint64_t    benz_itu_h26xbs_read_ue(BENZ_ITU_H26XBS* bs){
    uint64_t v, K=0x0000000100000000ULL;
    unsigned a, b, c;
    
    v = bs->sreg;
    a = benz_clz64(v|K);
    b = 2*a+1;
    c = 63-a;
    bs->sregoff += b;
    bs->sreg     = v << b;
    bs->errmask |= v>=K ? 0 : BENZ_H26XBS_ERR_CORRUPT;
    
    return ((v << a) >> c) - 1;
}
BENZINA_PUBLIC BENZINA_INLINE int64_t     benz_itu_h26xbs_read_se(BENZ_ITU_H26XBS* bs){
    uint64_t v = benz_itu_h26xbs_read_ue(bs);
    v = v&1 ? v+1 : -v;
    return (int64_t)v>>1;
}

/**
 * @brief Skip n bits, an unsigned or signed Exp-Golomb value, or until
 *        byte-aligned.
 * 
 * @param [in]  bs  Bitstream Parser
 * @param [in]  n   Number of bits to be skipped.
 */

BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_xn(BENZ_ITU_H26XBS* bs, unsigned n){
    bs->sreg   <<= n;
    bs->sregoff += n;
}
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_skip_xe(BENZ_ITU_H26XBS* bs){
    uint64_t v, K=0x0000000100000000ULL;
    unsigned a, b;
    
    v = bs->sreg;
    a = benz_clz64(v|K);
    b = 2*a+1;
    bs->sreg     = v << b;
    bs->sregoff += b;
    bs->errmask |= v>=K ? 0 : BENZ_H26XBS_ERR_CORRUPT;
}
BENZINA_PUBLIC BENZINA_INLINE void        benz_itu_h26xbs_realign(BENZ_ITU_H26XBS* bs){
    benz_itu_h26xbs_skip_xn(bs, -bs->sregoff & 7);
}

/**
 * @brief Check if bitstream parser is at end of stream.
 * 
 * @param [in]  bs  Bitstream Parser
 * @return !0 if at EOS, 0 otherwise.
 */

BENZINA_PUBLIC BENZINA_INLINE int         benz_itu_h26xbs_eos    (BENZ_ITU_H26XBS* bs){
    return bs->nalulen==0 && bs->sregoff >= bs->headoff;
}

/**
 * @brief Check if bitstream parser is in an error condition.
 * 
 * @param [in]  bs  Bitstream Parser
 * @return 0 if not in error, !0 otherwise.
 */

BENZINA_PUBLIC BENZINA_INLINE uint32_t    benz_itu_h26xbs_err    (BENZ_ITU_H26XBS* bs){
    bs->errmask |= bs->sregoff <= bs->tailoff+64 ? 0 : BENZ_H26XBS_ERR_OVERREAD;
    bs->errmask |= bs->sregoff <= bs->headoff    ? 0 : BENZ_H26XBS_ERR_OVERREAD;
    return bs->errmask;
}

/**
 * @brief Mark bitstream corrupt.
 * 
 * @param [in]  bs  Bitstream Parser
 * @return Error mask.
 */

BENZINA_PUBLIC BENZINA_INLINE uint32_t    benz_itu_h26xbs_markcor(BENZ_ITU_H26XBS* bs){
    return bs->errmask |= BENZ_H26XBS_ERR_CORRUPT;
}

/**
 * @brief Heavyweight Fill of RBSP Buffer and Shift Register.
 * 
 * Performs heavy-duty checks, and refills the RBSP ring buffer. Expected to be
 * called every 64-128 bytes or so, and refill the ring buffer by a similar
 * amount.
 * 
 * Idempotent. Safe.
 * 
 * @param [in]  bs  Bitstream Parser
 * @return 0
 */

BENZINA_PUBLIC                void        benz_itu_h26xbs_bigfill(BENZ_ITU_H26XBS* bs);

/**
 * @brief Heavyweight Skip of bits in RBSP buffer and shift register.
 * 
 * Performs heavy-duty checks, skips given number of bits, then refills the
 * RBSP buffer.
 * 
 * Safe.
 * 
 * @param [in]  bs  Bitstream Parser
 * @return 0
 */

BENZINA_PUBLIC                void        benz_itu_h26xbs_bigskip(BENZ_ITU_H26XBS* bs, uint64_t n);



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

