/* Includes */
#include "benzina/itu/h265.h"
#include "benzina/itu/h26x.h"



/* Defines */



/* Extern Inline Function Definitions */
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_check_header(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_get_type(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_vcl(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_idr(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_idr_n_lp(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_vps(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_sps(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_pps(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_aud(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_eos(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_eob(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_fd(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_itu_h265_nalu_is_sei(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_itu_h265_nalu_slice_get_pps_id(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_itu_h265_nalu_pps_get_pps_id(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_itu_h265_nalu_pps_get_sps_id(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_itu_h265_nalu_sps_get_vps_id(const BENZ_ITU_H265_NALU* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_itu_h265_nalu_vps_get_vps_id(const BENZ_ITU_H265_NALU* p);



/* Static Function Definitions */
BENZINA_STATIC inline                int            benz_itu_h265_epb_present(const void* p){
    const uint8_t* b = (const uint8_t*)p;
    return b[0] == 3 && b[-1] == 0 && b[-2] == 0;
}
BENZINA_STATIC                       const uint8_t* benz_itu_h265_skipbytes(const void* p, uint64_t off){
    const uint8_t* b = (const uint8_t*)p;
    int            epb;
    
    while(off >= 3){
        if(*b > 3){
            off -= 3;
            b   += 3;
        }else{
            off -= !benz_itu_h265_epb_present(b++);
        }
    }
    
    epb = 0;
    do{
        off -= epb;
        epb  = benz_itu_h265_epb_present(b);
        b   += epb;
    }while(off > 0);
    
    return b;
}


/* Public Function Definitions */
BENZINA_PUBLIC                       uint64_t    benz_itu_h265_nalu_sps_get_sps_id(const BENZ_ITU_H265_NALU* p){
    const uint8_t* b = (const uint8_t*)p;
    unsigned       sps_max_sub_layers_minus1;
    uint64_t       present_flags;
    int            cnt_profile_present;
    int            cnt_level_present;
    
    
    b += 2;
    sps_max_sub_layers_minus1 = (*b >> 1) & 7;
    b  = benz_itu_h265_skipbytes(b, 13);
    if(sps_max_sub_layers_minus1){
        present_flags       = (uint64_t)*b++ << 8;
        b                  += benz_itu_h265_epb_present(b);
        present_flags      |= (uint64_t)*b++;
        b                  += benz_itu_h265_epb_present(b);
        cnt_profile_present = benz_popcnt64(present_flags & 0xAAA8U);
        cnt_level_present   = benz_popcnt64(present_flags & 0x5554U);
        b                   = benz_itu_h265_skipbytes(b, cnt_profile_present*11+
                                                         cnt_level_present);
    }
    
    return benz_ueto64((uint64_t)*b << 56);
}

