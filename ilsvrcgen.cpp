/**
 * ILSVRC Generator.
 */

/* Includes */
#include <unistd.h>
#include <dirent.h>
#include <hdf5.h>
#include <jpeglib.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



/* Defines */
#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif


/* Implementation */

/**************************************************************
 ***         PRNG based on PCG XSH RR 64/32 (LCG)           ***
 **************************************************************/
/* Forward Declarations */
static       uint32_t pcgRor32 (uint32_t x, uint32_t n) UNUSED;
static       void     pcgSeed  (uint64_t seed)          UNUSED;
static       uint32_t pcgRand  (void)                   UNUSED;
static       double   pcgRand01(void)                   UNUSED;
/* Definitions */
static       uint64_t pcgS =                   1;/* State */
static const uint64_t pcgM = 6364136223846793005;/* Multiplier */
static const uint64_t pcgA = 1442695040888963407;/* Addend */
static       uint32_t pcgRor32 (uint32_t x, uint32_t n){
	return (n &= 0x1F) ? x>>n | x<<(32-n) : x;
}
static       void     pcgSeed  (uint64_t seed){
	pcgS = seed;
}
static       uint32_t pcgRand  (void){
	pcgS = pcgS*pcgM + pcgA;
	
	/**
	 * PCG does something akin to an unbalanced Feistel round to blind the LCG
	 * state:
	 * 
	 * The rightmost 59 bits are involved in an xorshift by 18.
	 * The leftmost   5 bits select a rotation of the 32 bits 58:27.
	 */
	
	return pcgRor32((pcgS^(pcgS>>18))>>27, pcgS>>59);
}
static       double   pcgRand01(void){
	uint64_t u = pcgRand(), l = pcgRand();
	uint64_t x = u<<32 | l;
	return x * ldexp(1,-64);
}




/**************************************************************
 ***                    Dataset Creation                    ***
 **************************************************************/

/**
 * The dataset is structured as follows:
 * 
 * /
 *   data/
 *     y                        u8[1461406][5],                     align=16MB
 *        # [*][0]: Byte offset start, inclusive
 *        # [*][1]: Byte offset end,   exclusive
 *        # [*][2]: Class label (0-999)
 *        # [*][3]: Width  in pixels
 *        # [*][4]: Height in pixels
 *     x                        u1["length of concatenated JPEGs"], align=16G
 *     splits                   u8[3][2],                           align=8
 *        # [0][0]: Training   Start Index (      0)
 *        # [0][1]: Training   End   Index (1261405)
 *        # [1][0]: Validation Start Index (1261406)
 *        # [1][1]: Validation End   Index (1311405)
 *        # [2][0]: Test       Start Index (1311406)
 *        # [2][1]: Test       End   Index (1461405)
 *     c                        u8[1000][3],                        align=4K
 *        # [*][0]: Class count in train subset
 *        # [*][1]: Class count in val   subset
 *        # [*][2]: Class count in test  subset
 *     cNames                   str[1000],                          align=4K
 *        # [*]:    Class name
 *   src/
 *     imagenetgen.c            u1["length of this file"]
 */

typedef struct DSET_CTX{
	int         ret;
	DIR*        testDir;
	DIR*        trainDir;
	DIR*        valDir;
	FILE*       testGT;
	FILE*       valGT;
	const char* h5FileName;
	
	hid_t       h5File;
	hid_t       h5FileAPL;
	hid_t       h5Data;
	hid_t       h5Datay;
	hid_t       h5Datax;
	hid_t       h5Datasplits;
	hid_t       h5Src;
	hid_t       h5Srcimagenetgendotc;
} DSET_CTX;

/**
 * Allocate a dirent directory entry.
 */

struct dirent* allocDirent(int dirFd){
	long name_max = fpathconf(dirFd, _PC_NAME_MAX);
	if(name_max == -1){
		name_max = 255;
	}
	
	size_t off = (offsetof(struct dirent, d_name));
	size_t len = off + name_max + 1;
	return malloc(len);
}

/**
 * Opens the current directory and checks that its topology matches the root
 * directory of the dataset. That is to say, the current directory contains:
 * 
 * /
 *   ILSVRC2010_test_ground_truth.txt
 *   ILSVRC2010_validation_ground_truth.txt
 *   test/
 *   train/
 *   val/
 */

int   inDsetRootDir(DSET_CTX* ctx){
	ctx->testDir  = opendir("test/");
	ctx->trainDir = opendir("train/");
	ctx->valDir   = opendir("val/");
	ctx->testGT   = fopen("ILSVRC2010_test_ground_truth.txt", "r");
	ctx->valGT    = fopen("ILSVRC2010_validation_ground_truth.txt", "r");
	
	if(!ctx->testDir || !ctx->trainDir || !ctx->valDir ||
	   !ctx->testGT  || !ctx->valGT){
		return 0;
	}
	
	struct dirent* d = allocDirent(dirfd(ctx->testDir));
	
	return 0;
}

/**
 * Initialize HDF5
 */

int   init(DSET_CTX* ctx, int argc, char* argv[]){
	memset(ctx, 0, sizeof(*ctx));
	int    ret = -1, inRootDir = 0;
	herr_t err;
	
	/**
	 * Sanity check
	 */
	
	inRootDir = inDsetRootDir(ctx);
	if(argc != 2 || !inRootDir){
		if(argc != 2){
			printf("Please run this program with exactly 1 argument:\n"
			       "  - The name of the HDF5 file to create. It must not exist already.\n");
		}else if(!inRootDir){
			printf("Please run this program from the dataset root directory.\n");
		}
		ret = 1;
		return 0;
	}else{
		ctx->h5FileName = argv[1];
	}
	
	err = H5open();
	if(err<0){
		ret = 1;
		return 0;
	}
	
	return 0;
}

int   sanityCheck(DSET_CTX* ctx){
	return 0;
}

int   gatherStats(DSET_CTX* ctx){
	return 0;
}

/**
 * Open requested HDF5 file.
 */

int   createH5File(DSET_CTX* ctx){
	ctx->h5FileAPL = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_alignment(ctx->h5FileAPL, (hsize_t)16<<30, (hsize_t)16<<30);
	
	ctx->h5File  = H5Fcreate(ctx->h5FileName,
	                         H5F_ACC_RDWR|H5F_ACC_CREAT|H5F_ACC_EXCL,
	                         H5P_FILE_CREATE_DEFAULT, ctx->h5FileAPL);
	if(ctx->h5File<0){
		ctx->ret = 1;
		return 0;
	}
	
	return 1;
}

int   mmapH5File(DSET_CTX* ctx){
	return 0;
}

int   fillH5File(DSET_CTX* ctx){
	return 0;
}

int   msyncH5File(DSET_CTX* ctx){
	return 0;
}

/**
 * Cleanup.
 */

int   cleanup(DSET_CTX* ctx){
	H5Pclose(ctx->h5FileAPL);
	H5Fclose(ctx->h5File);
	  H5Gclose(ctx->h5Data);
	    H5Dclose(ctx->h5Datax);
	    H5Dclose(ctx->h5Datay);
	    H5Dclose(ctx->h5Datasplits);
	  H5Gclose(ctx->h5Src);
	    H5Dclose(ctx->h5Srcimagenetgendotc);
	
	if(ctx->testDir) {closedir(ctx->testDir);}
	if(ctx->trainDir){closedir(ctx->trainDir);}
	if(ctx->valDir)  {closedir(ctx->valDir);}
	if(ctx->testGT)  {fclose  (ctx->testGT);}
	if(ctx->valGT)   {fclose  (ctx->valGT);}
	
	return ctx->ret;
}

/**
 * Main
 */

int   main(int argc, char* argv[]){
	DSET_CTX STACKCTX, *ctx=&STACKCTX;
	
	if(init        (ctx, argc, argv) &&
	   sanityCheck (ctx)             &&
	   gatherStats (ctx)             &&
	   createH5File(ctx)             &&
	   mmapH5File  (ctx)             &&
	   fillH5File  (ctx)             &&
	   msyncH5File (ctx)){ctx->ret=0;}
	
	return cleanup(ctx);
}
