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
    
    tstmessageflush(stdout, "1..2\n");
    
    
    benz_itu_h26xbs_init(bs, NULL, 0);
    tstmessagetap(bs->naluptr == NULL, "init 1");
    tstmessagetap(bs->nalulen == 0,    "init 2");
}
