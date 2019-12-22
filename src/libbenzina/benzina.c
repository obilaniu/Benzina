/* Includes */
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "benzina/benzina-old.h"


/* Defines */



/* Data Structure Definitions */

/**
 * @brief Benzina Dataset.
 * 
 * A description of the dataset and its decoding parameters.
 */

struct BENZINA_DATASET{
	char*                 file;
	size_t                length;
};



/* Static Function Declarations */
BENZINA_STATIC void  benzinaInitOnce(void);



/* Global Variables & Constants. */
static pthread_once_t benzinaInitOnceControl = PTHREAD_ONCE_INIT;
static int            benzinaInitOnceStatus  = -1;



/* Static Function Definitions */

/**
 * Perform initialization of libbenzina exactly once.
 */

BENZINA_STATIC void  benzinaInitOnce(void){
	benzinaInitOnceStatus = 0;
}


/* Public Function Definitions */
int          benzinaInit                 (void){
	int    ret = pthread_once(&benzinaInitOnceControl, benzinaInitOnce);
	return ret < 0 ? ret : benzinaInitOnceStatus;
}

int          benzinaDatasetAlloc         (BENZINA_DATASET** ctx){
	return -!(*ctx = malloc(sizeof(**ctx)));
}

int          benzinaDatasetInit          (BENZINA_DATASET*  ctx, const char* file, uint64_t* length){
	int         ret=0, filefd=-1;
	
	
	/* Wipe previous contents. */
	memset(ctx, 0, sizeof(*ctx));
	
	/* Duplicate path to file. */
	if(!file){
		return -1;
	}
	ctx->file = strdup(file);
	if(!ctx->file){
		return -1;
	}
	
	/* Test the existence of the expected files in the dataset file. */
	filefd = open(ctx->file, O_RDONLY|O_CLOEXEC);
	if(filefd < 0){
		ret = -1;
		goto abortprobe;
	}
	
	ctx->length = *length;
	if(ctx->length <= 0){
		ret = -2;
		goto abortprobe;
	}
	
	
	/* Return. */
	exitprobe:
	close(filefd);
	return ret;
	
	
	/* Abort path. */
	abortprobe:
	free(ctx->file);
	ctx->file   = NULL;
	ctx->length = 0;
	goto exitprobe;
}

int          benzinaDatasetNew           (BENZINA_DATASET** ctx, const char* file, uint64_t* length){
	int ret = benzinaDatasetAlloc(ctx);
	return ret ? ret : benzinaDatasetInit(*ctx, file, length);
}

int          benzinaDatasetFini          (BENZINA_DATASET*  ctx){
	free(ctx->file);
	memset(ctx, 0, sizeof(*ctx));
	return 0;
}

int          benzinaDatasetFree          (BENZINA_DATASET*  ctx){
	if(ctx){
		benzinaDatasetFini(ctx);
		free(ctx);
	}
	return 0;
}

int          benzinaDatasetGetFile       (const BENZINA_DATASET*  ctx, const char** path){
	*path = ctx->file;
	return !*path;
}

int          benzinaDatasetGetLength     (const BENZINA_DATASET*  ctx, size_t* length){
	*length = ctx->length;
	return 0;
}
