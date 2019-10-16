/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <benzina/benzina.h>
#include <benzina/itu/h26x-bitstream.h>



/**
 * Defines
 */

#define tstmessageflush(fp, f, ...)               \
    do{                                           \
        fprintf((fp), (f), ## __VA_ARGS__);       \
        fflush((fp));                             \
    }while(0)
#define tstmessagetap(r, f, ...)                  \
    do{                                           \
        if((r)){                                  \
            fprintf(stdout, "ok ");               \
        }else{                                    \
            fprintf(stdout, "not ok ");           \
        }                                         \
        fprintf(stdout, (f), ## __VA_ARGS__);     \
        fprintf(stdout, "\n");                    \
        fflush(stdout);                           \
    }while(0)
#define tstmessage(fp, f, ...)                    \
    (fprintf((fp), (f), ## __VA_ARGS__))
#define tstmessageexit(code, fp, f, ...)          \
    do{                                           \
        tstmessageflush(fp, f, ## __VA_ARGS__);   \
        exit(code);                               \
    }while(0)



/**
 * Functions
 */



/**
 * Main
 */

int main(void){
    BENZ_H26XBS bs_STACK, *bs = &bs_STACK;
    
    
    tstmessageflush(stdout, "1..24\n");
    
    
    /* Null pointer, 0 bytes {} */
    benz_itu_h26xbs_init(bs, NULL, 0);
    tstmessagetap(bs->naluptr == NULL,      "null init, 0 bytes ptr");
    tstmessagetap(bs->nalulen == 0,         "null init, 0 bytes len");
    tstmessagetap(benz_itu_h26xbs_eos(bs),  "null init, 0 bytes EOS");
    
    /* Null pointer, >0 bytes *NULL */
    benz_itu_h26xbs_init(bs, NULL, 100);
    tstmessagetap(bs->naluptr == NULL,      "null init, >0 bytes ptr");
    tstmessagetap(bs->nalulen == 0,         "null init, >0 bytes len");
    tstmessagetap(benz_itu_h26xbs_eos(bs),  "null init, >0 bytes EOS");
    
    /* Non-null pointer, 0 bytes {} */
    benz_itu_h26xbs_init(bs, "", 0);
    tstmessagetap(bs->naluptr != NULL,      "non-null init, 0 bytes ptr");
    tstmessagetap(bs->nalulen == 0,         "non-null init, 0 bytes len");
    tstmessagetap(benz_itu_h26xbs_eos(bs),  "non-null init, 0 bytes EOS");
    
    /* Non-null pointer, 1 bytes {0x03} */
    benz_itu_h26xbs_init(bs, "\x03", 1);
    tstmessagetap(bs->naluptr != NULL,                  "non-null init, 1 bytes ptr");
    tstmessagetap(!benz_itu_h26xbs_eos(bs),             "non-null init, 1 bytes !EOS");
    tstmessagetap(!benz_itu_h26xbs_err(bs),             "non-null init, 1 bytes !ERR");
    tstmessagetap(benz_itu_h26xbs_read_1b(bs)    == 0,  "non-null init, 1 bytes, 1 bit read");
    tstmessagetap(benz_itu_h26xbs_read_un(bs, 2) == 0,  "non-null init, 1 bytes, 2 bit unsigned read");
    tstmessagetap(benz_itu_h26xbs_read_sn(bs, 3) == 0,  "non-null init, 1 bytes, 3 bit signed read");
    tstmessagetap(benz_itu_h26xbs_read_ue(bs)    == 0,  "non-null init, 1 bytes, unsigned Exp-Golomb");
    tstmessagetap(benz_itu_h26xbs_read_ue(bs)    == 0,  "non-null init, 1 bytes, signed Exp-Golomb");
    tstmessagetap(benz_itu_h26xbs_eos(bs),              "non-null init, 1 bytes EOS");
    tstmessagetap(!benz_itu_h26xbs_err(bs),             "non-null init, 1 bytes EOS+!ERR");
    benz_itu_h26xbs_skip_xn(bs, 5);
    tstmessagetap(bs->sregoff == bs->headoff+5,         "non-null init, 1 bytes, 5 bit overread");
    tstmessagetap(benz_itu_h26xbs_err(bs) ==
                  BENZ_H26XBS_ERR_OVERREAD,             "non-null init, 1 bytes, ERR=overread");
    
    /* Non-null pointer, 2 bytes {0x00, 0x03} */
    benz_itu_h26xbs_init(bs, "\x00\x03", 2);
    tstmessagetap(bs->naluptr != NULL, "non-null init, 2 bytes ptr");
    
    /* Non-null pointer, 3 bytes, EPB {0x00, 0x00, 0x03} */
    benz_itu_h26xbs_init(bs, "\x00\x00\x03", 3);
    tstmessagetap(bs->naluptr != NULL, "non-null init, 3 bytes, EPB ptr");
    
    /* Non-null pointer, 3 bytes, no EPB {0x00, 0x00, 0xFF} */
    benz_itu_h26xbs_init(bs, "\x00\x00\xFF", 3);
    tstmessagetap(bs->naluptr != NULL, "non-null init, 3 bytes, no EPB ptr");
}
