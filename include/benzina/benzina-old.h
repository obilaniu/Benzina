/* Include Guard */
#ifndef INCLUDE_BENZINA_BENZINA_H
#define INCLUDE_BENZINA_BENZINA_H

/**
 * Includes
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "benzina/bits.h"
#include "benzina/visibility.h"



/* Defines */



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declarations and Typedefs */
typedef struct BENZINA_BUF             BENZINA_BUF;
typedef struct BENZINA_DATASET         BENZINA_DATASET;

struct BENZINA_BUF{
    char*  buf;
    size_t off;
    size_t len;
    size_t maxLen;
};



/* Function Prototypes */

/**
 * @brief Benzina Initialization.
 * 
 * @return Zero if successful; Non-zero if not successful.
 */

BENZINA_PUBLIC int          benzinaInit                 (void);

/**
 * @brief BENZINA_DATASET operations.
 */

BENZINA_PUBLIC int          benzinaDatasetAlloc         (BENZINA_DATASET**         ctx);
BENZINA_PUBLIC int          benzinaDatasetInit          (BENZINA_DATASET*          ctx, const char*  file, uint64_t* length);
BENZINA_PUBLIC int          benzinaDatasetNew           (BENZINA_DATASET**         ctx, const char*  file, uint64_t* length);
BENZINA_PUBLIC int          benzinaDatasetFini          (BENZINA_DATASET*          ctx);
BENZINA_PUBLIC int          benzinaDatasetFree          (BENZINA_DATASET*          ctx);
BENZINA_PUBLIC int          benzinaDatasetGetFile       (const BENZINA_DATASET*    ctx, const char** path);
BENZINA_PUBLIC int          benzinaDatasetGetLength     (const BENZINA_DATASET*    ctx, size_t* length);

/**
 * @brief BENZINA_BUF (ProtoBuf) operations.
 */

BENZINA_PUBLIC int          benzinaBufInit       (BENZINA_BUF*  bbuf);
BENZINA_PUBLIC int          benzinaBufFini       (BENZINA_BUF*  bbuf);

BENZINA_PUBLIC int          benzinaBufEnsure     (BENZINA_BUF*  bbuf, size_t freeSpace);

BENZINA_PUBLIC int          benzinaBufSeek       (BENZINA_BUF*  bbuf,
                                                  ssize_t       off,
                                                  int           whence);
BENZINA_PUBLIC int          benzinaBufRead       (BENZINA_BUF*  bbuf,
                                                  char*         data,
                                                  size_t        len);
BENZINA_PUBLIC int          benzinaBufWrite      (BENZINA_BUF*  bbuf,
                                                  const char*   data,
                                                  size_t        len);
BENZINA_PUBLIC int          benzinaBufWriteFromFd(BENZINA_BUF*  bbuf,
                                                  int           fd,
                                                  size_t        len);

BENZINA_PUBLIC int          benzinaBufReadDelim  (BENZINA_BUF*  bbuf,
                                                  const char**  data,
                                                  size_t*       len);
BENZINA_PUBLIC int          benzinaBufReadStr    (BENZINA_BUF*  bbuf,
                                                  char**        str,
                                                  size_t*       len);
BENZINA_PUBLIC int          benzinaBufReadMsg    (BENZINA_BUF*  bbuf,
                                                  BENZINA_BUF*  msg);

BENZINA_PUBLIC int          benzinaBufReadvu64   (BENZINA_BUF*  bbuf, uint64_t* v);
BENZINA_PUBLIC int          benzinaBufReadvs64   (BENZINA_BUF*  bbuf, int64_t*  v);
BENZINA_PUBLIC int          benzinaBufReadvi64   (BENZINA_BUF*  bbuf, int64_t*  v);
BENZINA_PUBLIC int          benzinaBufReadvu32   (BENZINA_BUF*  bbuf, uint32_t* v);
BENZINA_PUBLIC int          benzinaBufReadvs32   (BENZINA_BUF*  bbuf, int32_t*  v);
BENZINA_PUBLIC int          benzinaBufReadvi32   (BENZINA_BUF*  bbuf, int32_t*  v);

BENZINA_PUBLIC int          benzinaBufReadfu64   (BENZINA_BUF*  bbuf, uint64_t* u);
BENZINA_PUBLIC int          benzinaBufReadfs64   (BENZINA_BUF*  bbuf, int64_t*  s);
BENZINA_PUBLIC int          benzinaBufReadff64   (BENZINA_BUF*  bbuf, double*   f);
BENZINA_PUBLIC int          benzinaBufReadfu32   (BENZINA_BUF*  bbuf, uint32_t* u);
BENZINA_PUBLIC int          benzinaBufReadfs32   (BENZINA_BUF*  bbuf, int32_t*  s);
BENZINA_PUBLIC int          benzinaBufReadff32   (BENZINA_BUF*  bbuf, float*    f);

BENZINA_PUBLIC int          benzinaBufReadEnum   (BENZINA_BUF*  bbuf, int32_t*  e);
BENZINA_PUBLIC int          benzinaBufReadBool   (BENZINA_BUF*  bbuf, int*      b);

BENZINA_PUBLIC int          benzinaBufReadTagW   (BENZINA_BUF*  bbuf,
                                                  uint32_t*     tag,
                                                  uint32_t*     wire);

BENZINA_PUBLIC int          benzinaBufReadSkip   (BENZINA_BUF*  bbuf,
                                                  uint32_t      wire);




/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

