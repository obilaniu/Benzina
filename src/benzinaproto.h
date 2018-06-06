/* Include Guard */
#ifndef SRC_BENZINAPROTO_H
#define SRC_BENZINAPROTO_H

/**
 * Includes
 */

#include <stdint.h>
#include "benzina/benzina.h"



/* Defines */




/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declarations and Typedefs */
typedef struct BENZINA_BUF BENZINA_BUF;

struct BENZINA_BUF{
	char*  buf;
	size_t off;
	size_t len;
	size_t maxLen;
};



/* Function Prototypes */
BENZINA_HIDDEN int          benzinaBufInit     (BENZINA_BUF*  bbuf);
BENZINA_HIDDEN int          benzinaBufFini     (BENZINA_BUF*  bbuf);

BENZINA_HIDDEN int          benzinaBufEnsure   (BENZINA_BUF*  bbuf, size_t freeSpace);

/* Raw Read/Write calls. */
BENZINA_HIDDEN int          benzinaBufRead     (BENZINA_BUF*  bbuf,
                                                char*         data,
                                                size_t        len);
BENZINA_HIDDEN int          benzinaBufWrite    (BENZINA_BUF*  bbuf,
                                                const char*   data,
                                                size_t        len);

/* Protobuf */
/* Protobuf Read */
BENZINA_HIDDEN int          benzinaBufReadLDL  (BENZINA_BUF*  bbuf,
                                                const char**  data,
                                                size_t*       len);
BENZINA_HIDDEN int          benzinaBufReadStr  (BENZINA_BUF*  bbuf,
                                                char**        str,
                                                size_t*       len);
BENZINA_HIDDEN int          benzinaBufReadMsg  (BENZINA_BUF*  bbuf,
                                                BENZINA_BUF*  msg);

BENZINA_HIDDEN int          benzinaBufReadvu64 (BENZINA_BUF*  bbuf, uint64_t* v);
BENZINA_HIDDEN int          benzinaBufReadvs64 (BENZINA_BUF*  bbuf, int64_t*  v);
BENZINA_HIDDEN int          benzinaBufReadvi64 (BENZINA_BUF*  bbuf, int64_t*  v);
BENZINA_HIDDEN int          benzinaBufReadvu32 (BENZINA_BUF*  bbuf, uint32_t* v);
BENZINA_HIDDEN int          benzinaBufReadvs32 (BENZINA_BUF*  bbuf, int32_t*  v);
BENZINA_HIDDEN int          benzinaBufReadvi32 (BENZINA_BUF*  bbuf, int32_t*  v);

BENZINA_HIDDEN int          benzinaBufReadfu64 (BENZINA_BUF*  bbuf, uint64_t* u);
BENZINA_HIDDEN int          benzinaBufReadfs64 (BENZINA_BUF*  bbuf, int64_t*  s);
BENZINA_HIDDEN int          benzinaBufReadff64 (BENZINA_BUF*  bbuf, double*   f);
BENZINA_HIDDEN int          benzinaBufReadfu32 (BENZINA_BUF*  bbuf, uint32_t* u);
BENZINA_HIDDEN int          benzinaBufReadfs32 (BENZINA_BUF*  bbuf, int32_t*  s);
BENZINA_HIDDEN int          benzinaBufReadff32 (BENZINA_BUF*  bbuf, float*    f);

BENZINA_HIDDEN int          benzinaBufReadEnum (BENZINA_BUF*  bbuf, int32_t*  e);
BENZINA_HIDDEN int          benzinaBufReadBool (BENZINA_BUF*  bbuf, int*      b);

BENZINA_HIDDEN int          benzinaBufReadTagW (BENZINA_BUF*  bbuf,
                                                uint32_t*     tag,
                                                uint32_t*     wire);

/* Protobuf Write */
BENZINA_HIDDEN int          benzinaBufWriteLDL (BENZINA_BUF*  bbuf,
                                                const char*   data,
                                                size_t        len);
BENZINA_HIDDEN int          benzinaBufWriteStr (BENZINA_BUF*  bbuf, const char*        str);
BENZINA_HIDDEN int          benzinaBufWriteMsg (BENZINA_BUF*  bbuf,
                                                const BENZINA_BUF* msg,
                                                size_t        len);

BENZINA_HIDDEN int          benzinaBufWritevu64(BENZINA_BUF*  bbuf, uint64_t  v);
BENZINA_HIDDEN int          benzinaBufWritevs64(BENZINA_BUF*  bbuf, int64_t   v);
BENZINA_HIDDEN int          benzinaBufWritevi64(BENZINA_BUF*  bbuf, int64_t   v);
BENZINA_HIDDEN int          benzinaBufWritevu32(BENZINA_BUF*  bbuf, uint32_t  v);
BENZINA_HIDDEN int          benzinaBufWritevs32(BENZINA_BUF*  bbuf, int32_t   v);
BENZINA_HIDDEN int          benzinaBufWritevi32(BENZINA_BUF*  bbuf, int32_t   v);

BENZINA_HIDDEN int          benzinaBufWritefu64(BENZINA_BUF*  bbuf, uint64_t  u);
BENZINA_HIDDEN int          benzinaBufWritefs64(BENZINA_BUF*  bbuf, int64_t   s);
BENZINA_HIDDEN int          benzinaBufWriteff64(BENZINA_BUF*  bbuf, double    f);
BENZINA_HIDDEN int          benzinaBufWritefu32(BENZINA_BUF*  bbuf, uint32_t  u);
BENZINA_HIDDEN int          benzinaBufWritefs32(BENZINA_BUF*  bbuf, int32_t   s);
BENZINA_HIDDEN int          benzinaBufWriteff32(BENZINA_BUF*  bbuf, float     f);

BENZINA_HIDDEN int          benzinaBufWriteEnum(BENZINA_BUF*  bbuf, int32_t   e);
BENZINA_HIDDEN int          benzinaBufWriteBool(BENZINA_BUF*  bbuf, uint64_t  b);

BENZINA_HIDDEN int          benzinaBufWriteTagW(BENZINA_BUF*  bbuf,
                                                uint32_t      tag,
                                                uint32_t      wire);




/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

