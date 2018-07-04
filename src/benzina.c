/* Includes */
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "benzina/benzina.h"


/* Defines */



/* Data Structure Definitions */

/**
 * @brief Benzina Dataset.
 * 
 * A description of the dataset and its decoding parameters.
 */

struct BENZINA_DATASET{
	char*                 root;
	size_t                length;
	uint64_t*             lengths;
	uint64_t*             offsets;
	uint64_t              codedWidth;
	uint64_t              codedHeight;
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

int          benzinaDatasetInit          (BENZINA_DATASET*  ctx, const char* root){
	BENZINA_BUF bbuf = {0};
	struct stat s0, s1, s2, s3, s4;
	int         ret=0, dirfd=-1, fd1=-1, fd2=-1;
	size_t      bytesRead=0, bytesLeft=0, i;
	ssize_t     bytesChunk=0;
	uint32_t    tag, wire;
	
	
	/* Wipe previous contents. */
	memset(ctx, 0, sizeof(*ctx));
	
	/* Duplicate path to root. */
	if(!root){
		return -1;
	}
	ctx->root = strdup(root);
	if(!ctx->root){
		return -1;
	}
	
	/* Test the existence of the expected files in the dataset root. */
	dirfd = open(ctx->root, O_RDONLY|O_CLOEXEC|O_DIRECTORY);
	if(dirfd                                   < 0 ||
	   fstatat(dirfd, "data.bin",      &s0, 0) < 0 ||
	   fstatat(dirfd, "data.lengths",  &s1, 0) < 0 ||
	   fstatat(dirfd, "data.protobuf", &s2, 0) < 0 ||
	   fstatat(dirfd, "README.md",     &s3, 0) < 0 ||
	   fstatat(dirfd, "SHA256SUMS",    &s4, 0) < 0){
		ret = -1;
		goto abortprobe;
	}
	
	/**
	 * The data.lengths file is a simple array of uint64_t that defines how long
	 * each record in the data.bin file concatenation is. Because it is of fixed
	 * size per entry, we use it as the canonical source of information for the
	 * length of the dataset.
	 */
	
	ctx->length  = s1.st_size/8;
	ctx->lengths = malloc(s1.st_size);
	ctx->offsets = malloc(s1.st_size);
	if(!ctx->lengths || !ctx->offsets){
		ret = -2;
		goto abortprobe;
	}
	
	/* Read fully data.lengths into our private buffer. */
	fd1 = openat(dirfd, "data.lengths", O_RDONLY|O_CLOEXEC);
	if(fd1 < 0){
		ret = -1;
		goto abortprobe;
	}
	for(bytesLeft=s1.st_size, bytesRead=0; bytesLeft>0;){
		bytesChunk = pread(fd1, (char*)ctx->lengths+bytesRead, bytesLeft, bytesRead);
		if (bytesChunk <= 0){
			/* We should not end up at EOF with bytesLeft > 0. */
			ret = -3;
			goto abortprobe;
		}else{
			bytesRead += bytesChunk;
			bytesLeft -= bytesChunk;
		}
	}
	
	/**
	 * Simultaneously byte-transpose to native endianness and compute cumulative
	 * offset.
	 */
	
	uint64_t cumulative=0, rawLength=0;
	for(i=0;i<ctx->length;i++){
		rawLength       = ctx->lengths[i];
		#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
		rawLength       = __builtin_bswap64(rawLength);
		ctx->lengths[i] = rawLength;
		#endif
		ctx->offsets[i] = cumulative;
		cumulative     += rawLength;
	}
	
	/* Read ProtoBuf dataset description. */
	if(benzinaBufInit  (&bbuf)             < 0 ||
	   benzinaBufEnsure(&bbuf, s2.st_size) < 0){
		ret = -1;
		goto abortprobe;
	}
	fd2 = openat(dirfd, "data.protobuf", O_RDONLY|O_CLOEXEC);
	if(fd2 < 0 || benzinaBufWriteFromFd(&bbuf, fd2, s2.st_size) != 0){
		ret = -1;
		goto abortprobe;
	}
	bbuf.off = 0;
	while(benzinaBufReadTagW(&bbuf, &tag, &wire) == 0){
		switch(tag){
			case 33554432: benzinaBufReadvu64(&bbuf, &ctx->codedWidth);  break;
			case 33554433: benzinaBufReadvu64(&bbuf, &ctx->codedHeight); break;
			default:       benzinaBufReadSkip(&bbuf, wire);              break;
		}
	}
	
	
	/* Return. */
	exitprobe:
	benzinaBufFini(&bbuf);
	close(dirfd);
	close(fd1);
	close(fd2);
	return ret;
	
	
	/* Abort path. */
	abortprobe:
	free(ctx->root);
	free(ctx->lengths);
	free(ctx->offsets);
	ctx->root    = NULL;
	ctx->lengths = NULL;
	ctx->offsets = NULL;
	goto exitprobe;
}

int          benzinaDatasetNew           (BENZINA_DATASET** ctx, const char* root){
	int ret = benzinaDatasetAlloc(ctx);
	return ret ? ret : benzinaDatasetInit(*ctx, root);
}

int          benzinaDatasetFini          (BENZINA_DATASET*  ctx){
	free(ctx->root);
	free(ctx->lengths);
	free(ctx->offsets);
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

int          benzinaDatasetGetRoot       (const BENZINA_DATASET*  ctx, const char** path){
	*path = ctx->root;
	return !*path;
}

int          benzinaDatasetGetLength     (const BENZINA_DATASET*  ctx, size_t* length){
	*length = ctx->length;
	return 0;
}

int          benzinaDatasetGetShape      (const BENZINA_DATASET*  ctx, size_t* w, size_t* h){
	*w = ctx->codedWidth;
	*h = ctx->codedHeight;
	return 0;
}

int          benzinaDatasetGetElement    (const BENZINA_DATASET*  ctx,
                                          size_t                  i,
                                          size_t*                 off,
                                          size_t*                 len){
	if(!ctx->offsets || !ctx->lengths || i>=ctx->length){
		*off = -1;
		*len = -1;
		return -1;
	}else{
		*off = ctx->offsets[i];
		*len = ctx->lengths[i];
		return 0;
	}
}

