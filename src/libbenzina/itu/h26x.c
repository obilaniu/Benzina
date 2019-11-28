/* Includes */
#include "benzina/itu/h26x.h"
#if 0
# if defined(__x86_64__) || defined(__i386__) || defined(_M_AMD64) || defined(_M_IX86)
#  include <immintrin.h>
# endif
#endif



/* Defines */




/* Extern Inline Function Definitions */
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_fill57b(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_fill64b(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_fill8B (BENZ_ITU_H26XBS* bs);

BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h26xbs_read_1b(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_itu_h26xbs_read_un(BENZ_ITU_H26XBS* bs, unsigned n);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int64_t     benz_itu_h26xbs_read_sn(BENZ_ITU_H26XBS* bs, unsigned n);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_itu_h26xbs_read_ue(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int64_t     benz_itu_h26xbs_read_se(BENZ_ITU_H26XBS* bs);

BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_skip_xn(BENZ_ITU_H26XBS* bs, unsigned n);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_skip_xe(BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_realign(BENZ_ITU_H26XBS* bs);

BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h26xbs_eos    (BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_itu_h26xbs_err    (BENZ_ITU_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_itu_h26xbs_markcor(BENZ_ITU_H26XBS* bs);



/* Target-specific Function Definitions */

/**
 * TODO: I dislike the organization and enabling of this code. I would prefer
 * that it be moved out to an arch/ folder.
 */

#if   0 && __SSE2__
BENZINA_STATIC void        benz_itu_h26xbs_epbstrip_sse2(BENZ_H26XBS* bs){
    while((int64_t)bs->nalulen > 0 && bs->headoff <= 1920){
        __m128i srcm2 = _mm_loadu_si128((void*)&bs->naluptr[-2]);
        __m128i srcm1 = _mm_loadu_si128((void*)&bs->naluptr[-1]);
        __m128i srcm0 = _mm_loadu_si128((void*)&bs->naluptr[-0]);
        
        __m128i srcd2 = _mm_cmpeq_epi8(srcm2, _mm_setzero_si128());
        __m128i srcd1 = _mm_cmpeq_epi8(srcm1, _mm_setzero_si128());
        __m128i srcd0 = _mm_cmpeq_epi8(srcm0, _mm_set1_epi8(3));
        __m128i anded = _mm_and_si128(_mm_and_si128(srcd2, srcd1), srcd0);
        __m128i ored  = _mm_or_si128(anded, _mm_slli_si128(anded, 1));
        __m128i saded = _mm_sad_epu8(_mm_sub_epi8(_mm_setzero_si128(), anded),
                                     _mm_setzero_si128());
        
        __m128i csum  = ored;
        csum          = _mm_add_epi8(csum, _mm_slli_si128(csum, 1));
        csum          = _mm_add_epi8(csum, _mm_slli_si128(csum, 2));
        csum          = _mm_add_epi8(csum, _mm_slli_si128(csum, 4));
        csum          = _mm_add_epi8(csum, _mm_slli_si128(csum, 8));
        
        __m128i insr5 = _mm_cmpeq_epi8(csum, _mm_set1_epi8(-10));
        __m128i insr4 = _mm_cmpeq_epi8(csum, _mm_set1_epi8(- 8));
        __m128i insr3 = _mm_cmpeq_epi8(csum, _mm_set1_epi8(- 6));
        __m128i insr2 = _mm_cmpeq_epi8(csum, _mm_set1_epi8(- 4));
        __m128i insr1 = _mm_cmpeq_epi8(csum, _mm_set1_epi8(- 2));
        __m128i insr0 = _mm_cmpeq_epi8(csum, _mm_setzero_si128());
        
        __m128i dsts5 = _mm_srli_si128(_mm_and_si128(srcm0, insr5), 5);
        __m128i dsts4 = _mm_srli_si128(_mm_and_si128(srcm0, insr4), 4);
        __m128i dsts3 = _mm_srli_si128(_mm_and_si128(srcm0, insr3), 3);
        __m128i dsts2 = _mm_srli_si128(_mm_and_si128(srcm0, insr2), 2);
        __m128i dsts1 = _mm_srli_si128(_mm_and_si128(srcm0, insr1), 1);
        __m128i dsts0 =                _mm_and_si128(srcm0, insr0);
        __m128i dstwb = _mm_or_si128(dsts5,
                        _mm_or_si128(dsts4,
                        _mm_or_si128(dsts3,
                        _mm_or_si128(dsts2,
                        _mm_or_si128(dsts1,
                                     dsts0)))));
        
        _mm_storeu_si128((void*)&bs->rbsp[bs->headoff>>3], dstwb);
        
        size_t blocklen = 16-_mm_extract_epi16(saded, 0)-_mm_extract_epi16(saded, 4);
        bs->headoff += blocklen<<3;
        bs->nalulen -= 16;
        bs->naluptr += 16;
    }
}
#elif 0 && __SSSE3__
BENZINA_STATIC void        benz_itu_h26xbs_epbstrip_ssse3(BENZ_H26XBS* bs){
    while((int64_t)bs->nalulen > 0 && bs->headoff <= 1920){
        __m128i srcm2 = _mm_loadu_si128((void*)&bs->naluptr[-2]);
        __m128i srcm1 = _mm_loadu_si128((void*)&bs->naluptr[-1]);
        __m128i srcm0 = _mm_loadu_si128((void*)&bs->naluptr[-0]);
        
        __m128i srcd2 = _mm_cmpeq_epi8(srcm2, _mm_setzero_si128());
        __m128i srcd1 = _mm_cmpeq_epi8(srcm1, _mm_setzero_si128());
        __m128i srcd0 = _mm_cmpeq_epi8(srcm0, _mm_set1_epi8(3));
        __m128i anded = _mm_and_si128(_mm_and_si128(srcd2, srcd1), srcd0);
        __m128i saded = _mm_sad_epu8(_mm_sub_epi8(_mm_setzero_si128(), anded),
                                     _mm_setzero_si128());
        
        __m128i csum  = anded;
        csum          = _mm_add_epi8(csum, _mm_slli_si128(csum, 1));
        csum          = _mm_add_epi8(csum, _mm_slli_si128(csum, 2));
        csum          = _mm_add_epi8(csum, _mm_slli_si128(csum, 4));
        csum          = _mm_add_epi8(csum, _mm_slli_si128(csum, 8));
        csum          = _mm_add_epi8(csum, _mm_set_epi8(127,126,125,124,
                                                        123,122,121,120,
                                                        119,118,117,116,
                                                        115,114,113,112));
        
        __m128i dstwb = _mm_shuffle_epi8(srcm0, csum);
        
        _mm_storeu_si128((void*)&bs->rbsp[bs->headoff>>3], dstwb);
        
        size_t blocklen = 16-_mm_extract_epi16(saded, 0)-_mm_extract_epi16(saded, 4);
        bs->headoff += blocklen<<3;
        bs->nalulen -= 16;
        bs->naluptr += 16;
    }
}
#else
BENZINA_STATIC void        benz_itu_h26xbs_epbstrip_default(BENZ_ITU_H26XBS* bs){
    while(bs->nalulen > 0 && bs->headoff <= 2040){
        if(bs->naluptr[0]!=3 || bs->naluptr[-1]!=0 || bs->naluptr[-2]!=0){
            bs->rbsp[bs->headoff>>3] = *bs->naluptr;
            bs->headoff += 8;
        }
        bs->naluptr++;
        bs->nalulen--;
    }
}
#endif



/* Static Function Definitions */
BENZINA_STATIC void        benz_itu_h26xbs_epbstrip(BENZ_ITU_H26XBS* bs){
    #if   0 && __SSE2__
    benz_itu_h26xbs_epbstrip_sse2(bs);
    #elif 0 && __SSSE3__
    benz_itu_h26xbs_epbstrip_ssse3(bs);
    #else
    benz_itu_h26xbs_epbstrip_default(bs);
    #endif
}
BENZINA_STATIC void        benz_itu_h26xbs_markeof(BENZ_ITU_H26XBS* bs){
    bs->tailoff  = 0;
    bs->headoff  = 0;
    bs->sregoff  = 0;
    bs->naluptr += bs->nalulen;
    bs->nalulen  = 0;
    memset(bs->rbsp, 0, sizeof(bs->rbsp));
}



/* Function Definitions */
BENZINA_PUBLIC void        benz_itu_h26xbs_init(BENZ_ITU_H26XBS* bs,
                                                const void*  nalu,
                                                size_t       nalubytelen){
    bs->sreg    = 0;
    bs->errmask = 0;
    bs->headoff = 0;
    bs->tailoff = 0;
    bs->sregoff = 0;
    bs->naluptr = (uint8_t*)nalu;
    bs->nalulen = nalu ? (uint64_t)nalubytelen : 0;
    
    /**
     * Carefully handle NALUs in a way dependent upon its length.
     * The length-0,1,2 NALUs must be handled specially.
     */
    
    if      (bs->nalulen == 0){
        /* Empty NALUs are easiest to handle, we're done already. */
    }else if(bs->nalulen <= 2){
        /**
         * Tiny NALUs <2 bytes can be handled specially, since they cannot
         * have EPBs.
         */
        
        memcpy(bs->rbsp, bs->naluptr, bs->nalulen);
        bs->headoff  = bs->nalulen << 3;
        bs->tailoff  = 0;
        bs->sregoff  = 0;
        bs->naluptr += bs->nalulen;
        bs->nalulen -= bs->nalulen;
        benz_itu_h26xbs_fill8B(bs);
    }else{
        /**
         * Larger NALUs >2 bytes require a 2-byte priming process, since
         * bigfill can access memory at naluptr-2 bytes to determine
         * whether the current byte should be stripped.
         * 
         * Luckily, the first two bytes of a NALU cannot be EPBs, for much the
         * same reason as for tiny NALUs: Only the third byte and up of a NALU
         * can be an EPB.
         */
        
        memcpy(bs->rbsp+1536/8, bs->naluptr, 2);
        bs->headoff  = 1536+16;
        bs->tailoff  = 1536;
        bs->sregoff  = 1536;
        bs->naluptr += 2;
        bs->nalulen -= 2;
        benz_itu_h26xbs_bigfill(bs);
    }
}

BENZINA_PUBLIC void        benz_itu_h26xbs_bigfill(BENZ_ITU_H26XBS* bs){
    /**
     * This is the function with the job of sanitizing the state of the
     * bitstream reader at least somewhat, and wrapping the buffer.
     * 
     * We do not check for sregoff-tailoff > 64 on entry, as this can happen
     * because of a significant bit-skip. If this is not acceptable, the user
     * should invoke benz_itu_h26xbs_err() before this call.
     * 
     * We do check for sregoff exceeding headoff, which indicates failure to
     * call bigfill() often enough.
     */
    
    bs->errmask |= bs->sregoff <= bs->headoff ? 0 : BENZ_H26XBS_ERR_OVERREAD;
    if(bs->errmask){
        benz_itu_h26xbs_markeof(bs);
        return;
    }
    
    
    /**
     * If no NALU bytes are left to be filtered, can optimize.
     * The head offset not being in the 4th quarter indicates an impending
     * end-of-stream situation.
     */
    
    if(bs->headoff <= 1536)
        goto lightfill64exit;
    
    
    /**
     * Otherwise, there are NALU bytes left to filter. Try to make space for
     * the refill. If the shift register offset is in quarter:
     * 
     *   1), do nothing.
     *   2), move back by  64 bytes.
     *   3), move back by 128 bytes.
     *   4), move back by 192 bytes.
     *   beyond, then declare an error.
     */
    
    if      (bs->sregoff  <  512){
        goto lightfill64exit;
    }else if(bs->sregoff  < 1024){
        bs->sregoff -= 512;
        bs->headoff -= 512;
        memmove(bs->rbsp, bs->rbsp +  64, 192);
    }else if(bs->sregoff  < 1536){
        bs->sregoff -= 1024;
        bs->headoff -= 1024;
        memcpy (bs->rbsp, bs->rbsp + 128, 128);
    }else if(bs->sregoff  < 2048){
        bs->sregoff -= 1536;
        bs->headoff -= 1536;
        memcpy (bs->rbsp, bs->rbsp + 192,  64);
    }else if(bs->sregoff == 2048){
        bs->sregoff = 0;
        bs->headoff = 0;
    }else{
        bs->errmask |= BENZ_H26XBS_ERR_OVERREAD;
        benz_itu_h26xbs_markeof(bs);
        return;
    }
    bs->tailoff = bs->sregoff;
    
    
    /**
     * Top up RBSP buffer as much as convenient, thereby bringing the head
     * offset strictly inside the 4th quarter (>1536 bits).
     */
    
    benz_itu_h26xbs_epbstrip(bs);
    
    
    /**
     * Exit with a 64-bit shift-register refill.
     */
    
    lightfill64exit:
    benz_itu_h26xbs_fill64b(bs);
}

BENZINA_PUBLIC void        benz_itu_h26xbs_bigskip(BENZ_ITU_H26XBS* bs, uint64_t n){
    uint64_t m;
    int      overread = bs->sregoff > bs->headoff;
    
    if(overread){
        benz_itu_h26xbs_err(bs);
        return;
    }
    
    do{
        m  = bs->headoff-bs->sregoff;
        m  = m<n ? m : n;
        n -= m;
        benz_itu_h26xbs_skip_xn(bs, m);
        benz_itu_h26xbs_bigfill(bs);
    }while(n);
}

