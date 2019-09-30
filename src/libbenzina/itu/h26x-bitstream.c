/* Includes */
#include "benzina/itu/h26x-bitstream.h"



/* Defines */




/* Extern Inline Function Definitions */
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h26xbs_fill   (BENZ_H26XBS* bs);

BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h26xbs_read_1b(BENZ_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_itu_h26xbs_read_un(BENZ_H26XBS* bs, int n);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int64_t     benz_itu_h26xbs_read_sn(BENZ_H26XBS* bs, int n);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_itu_h26xbs_read_ue(BENZ_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int64_t     benz_itu_h26xbs_read_se(BENZ_H26XBS* bs);

BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_skip_nb(BENZ_H26XBS* bs, int n);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_skip_ue(BENZ_H26XBS* bs);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_skip_se(BENZ_H26XBS* bs);



/* Function Definitions */
BENZINA_PUBLIC int         benz_itu_h26xbs_init(BENZ_H26XBS* bs,
                                                const void*  nalu,
                                                size_t       nalubytelen){
    memset(bs->rbsp,  ~0, sizeof(bs->rbsp));
    bs->sreg     = 0;
    bs->sregrem  = 0;
    bs->rbsphead = (uint8_t*)&bs->rbsp[34];
    bs->rbsptail = 0;
    bs->naluptr  = (uint8_t*)nalu;
    bs->nalulen  = nalubytelen;
    return benz_itu_h26xbs_bigfill(bs);
}

BENZINA_PUBLIC int         benz_itu_h26xbs_bigfill(BENZ_H26XBS* bs){
    uint64_t copylen;
    
    if(bs->rbsphead >= &bs->rbsp[34]){
        bs->rbsphead -= 32;
        bs->rbsptail -= 32;
        memcpy(bs->rbsp, bs->rbsp+32, 32);
        
        if(bs->nalulen >= 16){
            
        }else{
            for(; bs->rbsptail < &bs->rbsp[48] && bs->nalulen; bs->nalulen--){
                *bs->rbsptail = *bs->naluptr++;
                if(memcmp(bs->rbsptail-2, "\x00\x00\x03", 3) != 0)
                    bs->rbsptail++;
            }
        }
    }
    
    return benz_itu_h26xbs_fill(bs);
}