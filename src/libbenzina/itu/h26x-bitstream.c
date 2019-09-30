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

BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_skip_xn(BENZ_H26XBS* bs, int n);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void        benz_itu_h26xbs_skip_xe(BENZ_H26XBS* bs);



/* Function Definitions */
BENZINA_PUBLIC int         benz_itu_h26xbs_init(BENZ_H26XBS* bs,
                                                const void*  nalu,
                                                size_t       nalubytelen){
    memset(bs->rbsp,  ~0, sizeof(bs->rbsp));
    bs->errmask   = 0;
    bs->rbsphead  = 0;
    bs->rbsptail  = 0;
    bs->sregshift = 0;
    bs->naluptr   = (uint8_t*)nalu;
    bs->sreg      = 0;
    bs->nalulen   = nalubytelen;
    return benz_itu_h26xbs_bigfill(bs);
}

BENZINA_PUBLIC int         benz_itu_h26xbs_bigfill(BENZ_H26XBS* bs){
    uint64_t copylen;
    
    if(bs->rbsphead >= 128*8){
        bs->rbsphead -= 128*8;
        bs->rbsptail -= 128*8;
        memcpy(bs->rbsp, bs->rbsp+128, sizeof(bs->rbsp)-128);
        
        if(bs->nalulen >= 128){
            
        }else{
            
        }
    }
    
    return benz_itu_h26xbs_fill(bs);
}

