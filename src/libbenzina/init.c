/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <lz4frame.h>
#include "internal.h"
#include "benzina/benzina.h"



/* Static Function Definitions */

/**
 * @brief Internally self-extract compressed sections using statically-linked LZ4.
 * 
 * Changes to this function must be coordinated with the linker script and
 * vice-versa. The magic symbols __lz4_decompress_array_{start,end} are defined
 * there.
 */

BENZINA_STATIC void benz_init_lz4(void){
    const BENZ_LZ4_ENTRY* p;
    LZ4F_errorCode_t            derr;
    LZ4F_decompressionContext_t dctx = NULL;
    LZ4F_decompressOptions_t    dopt = {.stableDst=1};
    size_t                      dO, dB, dM, sO, sB, sM;
    
    for(p=__lz4_cmd_array_start;
        p<__lz4_cmd_array_end; p++){
        if(p->src_end>p->src_start && p->dst_end>p->dst_start){
            sB = sM = p->src_end - p->src_start;
            dB = dM = p->dst_end - p->dst_start;
            dO = 0;
            sO = 0;
            
            for(derr = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
                dO< dM && sO< sM && !LZ4F_isError(derr);
                dO+=dB,   sO+=sB,   dB=dM-dO,   sB=sM-sO)
                derr = LZ4F_decompress(dctx, p->dst_start+dO, &dB,
                                             p->src_start+sO, &sB, &dopt);
            
            if(!LZ4F_isError(derr))
                derr = LZ4F_freeDecompressionContext(dctx);
            
            if(LZ4F_isError(derr))
                fprintf(stderr, "Fatal error during initialization in LZ4 self-extraction: %s\n",
                        LZ4F_getErrorName(derr)), fflush(stderr), abort();
        }
    }
}



/* Public Function Definitions */

/**
 * Initialize library.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_CONSTRUCTOR int benz_init(void){
    benz_init_lz4();
    return 0;
}
