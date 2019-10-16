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
    
    uint32_t a,b,c;
    tstmessageflush(stdout, "1..58\n");
    
    
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
    tstmessagetap(bs->naluptr != NULL,                     "non-null init, 1 bytes ptr");
    tstmessagetap(!benz_itu_h26xbs_eos(bs),                "non-null init, 1 bytes !EOS");
    tstmessagetap(!benz_itu_h26xbs_err(bs),                "non-null init, 1 bytes !ERR");
    tstmessagetap(benz_itu_h26xbs_read_1b(bs)    == 0,     "non-null init, 1 bytes, 1 bit read");
    tstmessagetap(benz_itu_h26xbs_read_un(bs, 2) == 0,     "non-null init, 1 bytes, 2 bit unsigned read");
    tstmessagetap(benz_itu_h26xbs_read_sn(bs, 3) == 0,     "non-null init, 1 bytes, 3 bit signed read");
    tstmessagetap(benz_itu_h26xbs_read_ue(bs)    == 0,     "non-null init, 1 bytes, unsigned Exp-Golomb");
    tstmessagetap(benz_itu_h26xbs_read_ue(bs)    == 0,     "non-null init, 1 bytes, signed Exp-Golomb");
    tstmessagetap(benz_itu_h26xbs_eos(bs),                 "non-null init, 1 bytes EOS");
    tstmessagetap(!benz_itu_h26xbs_err(bs),                "non-null init, 1 bytes EOS+!ERR");
    benz_itu_h26xbs_skip_xn(bs, 5);
    tstmessagetap(bs->sregoff == bs->headoff+5,            "non-null init, 1 bytes, 5 bit overread");
    tstmessagetap(benz_itu_h26xbs_err(bs) ==
                  BENZ_H26XBS_ERR_OVERREAD,                "non-null init, 1 bytes, ERR=overread");
    
    /* Non-null pointer, 2 bytes {0x00, 0x03} */
    benz_itu_h26xbs_init(bs, "\x00\x03", 2);
    tstmessagetap(bs->naluptr != NULL,                     "non-null init, 2 bytes ptr");
    tstmessagetap(benz_itu_h26xbs_read_un(bs, 16) == 3,    "non-null init, 2 bytes, 16 bit unsigned read");
    tstmessagetap(benz_itu_h26xbs_eos(bs),                 "non-null init, 2 bytes EOS");
    tstmessagetap(!benz_itu_h26xbs_err(bs),                "non-null init, 2 bytes EOS+!ERR");
    
    /* Non-null pointer, 3 bytes, EPB {0x00, 0x00, 0x03} */
    benz_itu_h26xbs_init(bs, "\x00\x00\x03", 3);
    tstmessagetap(bs->naluptr != NULL,                     "non-null init, 3 bytes, EPB ptr");
    tstmessagetap(benz_itu_h26xbs_read_sn(bs, 16) == 0,    "non-null init, 3 bytes, EPB 16 bit signed read");
    tstmessagetap(benz_itu_h26xbs_eos(bs),                 "non-null init, 3 bytes, EPB EOS");
    tstmessagetap(!benz_itu_h26xbs_err(bs),                "non-null init, 3 bytes, EPB EOS+!ERR");
    
    /* Non-null pointer, 3 bytes, no EPB {0x00, 0x00, 0xFF} */
    benz_itu_h26xbs_init(bs, "\x00\x00\xFF", 3);
    tstmessagetap(bs->naluptr != NULL,                     "non-null init, 3 bytes, no EPB ptr");
    tstmessagetap(benz_itu_h26xbs_read_sn(bs, 24) == 0xFF, "non-null init, 3 bytes, no EPB 24 bit signed read");
    tstmessagetap(benz_itu_h26xbs_eos(bs),                 "non-null init, 3 bytes, no EPB EOS");
    tstmessagetap(!benz_itu_h26xbs_err(bs),                "non-null init, 3 bytes, no EPB EOS+!ERR");
    
    /* Fills & Skips */
    benz_itu_h26xbs_init(bs, "\x01\x02\x03\x04\x05\x06\x07\x08"
                             "\x09\xAA\xB1\xC2\xD3\xE4\xF5\x10", 16);
    tstmessagetap(benz_itu_h26xbs_read_ue(bs) == 0x80,     "fill, 16 bytes, unsigned Exp-Golomb");
    tstmessagetap(bs->sregoff == bs->tailoff+15,           "fill, 16 bytes, unsigned Exp-Golomb 0x80 consumed 15 bits");
    benz_itu_h26xbs_fill57b(bs);
    tstmessagetap(bs->sregoff == bs->tailoff+7,            "fill, 16 bytes, fill57b refilled 57 bits");
    tstmessagetap(bs->sreg    == 0x0182028303840480ULL,    "fill, 16 bytes, shift register contains 57 bits");
    a = bs->sregoff;
    benz_itu_h26xbs_fill57b(bs);
    b = bs->sregoff;
    tstmessagetap(a           == b,                        "fill, 16 bytes, fill57b idempotent 1");
    tstmessagetap(bs->sreg    == 0x0182028303840480ULL,    "fill, 16 bytes, fill57b idempotent 2");
    benz_itu_h26xbs_fill64b(bs);
    tstmessagetap(bs->sregoff == bs->tailoff,              "fill, 16 bytes, fill64b refilled 64 bits");
    tstmessagetap(bs->sreg    == 0x01820283038404D5ULL,    "fill, 16 bytes, shift register contains 64 bits");
    a = bs->sregoff;
    benz_itu_h26xbs_fill64b(bs);
    b = bs->sregoff;
    tstmessagetap(a           == b,                        "fill, 16 bytes, fill64b idempotent 1");
    tstmessagetap(bs->sreg    == 0x01820283038404D5ULL,    "fill, 16 bytes, fill64b idempotent 2");
    a = bs->sregoff;
    benz_itu_h26xbs_bigfill(bs);
    tstmessagetap(bs->sreg    == 0x01820283038404D5ULL,    "fill, 16 bytes, bigfill refilled 64 bits");
    b = bs->sregoff;
    benz_itu_h26xbs_bigfill(bs);
    c = bs->sregoff;
    tstmessagetap(a == b && b == c,                        "fill, 16 bytes, bigfill idempotent 1");
    tstmessagetap(bs->sreg    == 0x01820283038404D5ULL,    "fill, 16 bytes, bigfill idempotent 2");
    a = bs->sregoff;
    benz_itu_h26xbs_realign(bs);
    b = bs->sregoff;
    benz_itu_h26xbs_realign(bs);
    c = bs->sregoff;
    tstmessagetap(a+1 == b && b%8 == 0,                    "fill, 16 bytes, realign on even byte boundary");
    tstmessagetap(b           == c,                        "fill, 16 bytes, realign idempotent");
    a = bs->sregoff;
    benz_itu_h26xbs_fill8B(bs);
    tstmessagetap(bs->sreg    == 0x03040506070809AAULL,    "fill, 16 bytes, fill8B refilled 8 bytes");
    a = bs->sregoff;
    benz_itu_h26xbs_fill8B(bs);
    b = bs->sregoff;
    tstmessagetap(a           == b,                        "fill, 16 bytes, fill8B idempotent 1");
    tstmessagetap(bs->sreg    == 0x03040506070809AAULL,    "fill, 16 bytes, fill8B idempotent 2");
    a = bs->sregoff;
    benz_itu_h26xbs_skip_xn(bs, 8);
    b = bs->sregoff;
    tstmessagetap(a+8         == b,                        "fill, 16 bytes, skipped 8 bits");
    tstmessagetap(bs->sreg    == 0x040506070809AA00ULL,    "fill, 16 bytes, consumed 8 bits");
    a = bs->sregoff;
    benz_itu_h26xbs_bigskip(bs, 0);
    b = bs->sregoff;
    tstmessagetap(a           == b,                        "fill, 16 bytes, bigskip skipped 0 bits");
    tstmessagetap(bs->sreg    == 0x040506070809AAB1ULL,    "fill, 16 bytes, bigskip refilled 64 bits");
    a = bs->sregoff;
    benz_itu_h26xbs_bigskip(bs, 104);
    b = bs->sregoff;
    tstmessagetap(a+104       == b,                        "fill, 16 bytes, bigskip skipped 104 bits");
    tstmessagetap(benz_itu_h26xbs_eos(bs),                 "fill, 16 bytes, EOS");
    tstmessagetap(!benz_itu_h26xbs_err(bs),                "fill, 16 bytes, !ERR");
    
    /* Exit. */
    return 0;
}
