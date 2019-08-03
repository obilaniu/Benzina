/* Includes */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "benzina/benzina-old.h"





/* Function Definitions */
BENZINA_PUBLIC int          benzinaBufInit       (BENZINA_BUF*  bbuf){
	bbuf->buf    = NULL;
	bbuf->off    = 0;
	bbuf->len    = 0;
	bbuf->maxLen = 0;
	return 0;
}
BENZINA_PUBLIC int          benzinaBufFini       (BENZINA_BUF*  bbuf){
	if(bbuf){
		free(bbuf->buf);
		bbuf->buf = NULL;
		memset(bbuf, 0, sizeof(*bbuf));
	}
	return 0;
}

BENZINA_PUBLIC int          benzinaBufEnsure     (BENZINA_BUF*  bbuf, size_t freeSpace){
	size_t newMaxLen = bbuf->off+freeSpace;
	void*  newBuf    = bbuf->buf;
	
	if(newMaxLen > bbuf->maxLen){
		newMaxLen = newMaxLen > 2*bbuf->maxLen ? newMaxLen : 2*bbuf->maxLen;
		newBuf    = realloc(bbuf->buf, newMaxLen);
		if(newBuf){
			bbuf->buf    = newBuf;
			bbuf->maxLen = newMaxLen;
		}
	}
	
	return !newBuf;
}

/* Raw Seek/Read/Write calls. */
BENZINA_PUBLIC int          benzinaBufSeek       (BENZINA_BUF*  bbuf,
                                                  ssize_t       off,
                                                  int           whence){
	switch(whence){
		case SEEK_SET:
			if(off < 0 || (size_t)off > bbuf->len){
				return -1;
			}else{
				bbuf->off = off;
				return 0;
			}
		case SEEK_CUR:
			if(((off < 0) && ((size_t)-off > bbuf->off            )) ||
			   ((off > 0) && ((size_t)+off > (bbuf->len-bbuf->off)))){
				return -1;
			}else{
				bbuf->off += off;
				return 0;
			}
		break;
		case SEEK_END:
			if(off > 0 || (size_t)-off > bbuf->len){
				return -1;
			}else{
				bbuf->off = bbuf->len+off;
				return 0;
			}
		break;
		default:
			return -1;
	}
}
BENZINA_PUBLIC int          benzinaBufRead       (BENZINA_BUF*  bbuf,
                                                  char*         data,
                                                  size_t        len){
	if(len > bbuf->len-bbuf->off){
		return 1;
	}else{
		memcpy(data, bbuf->buf+bbuf->off, len);
		bbuf->off += len;
		return 0;
	}
}
BENZINA_PUBLIC int          benzinaBufWrite      (BENZINA_BUF*  bbuf,
                                                  const char*   data,
                                                  size_t        len){
	int ret;
	
	ret = benzinaBufEnsure(bbuf, len);
	if(ret){
		return ret;
	}else{
		if(bbuf->off+len > bbuf->len){
			bbuf->len = bbuf->off+len;
		}
		memcpy(bbuf->buf+bbuf->off, data, len);
		bbuf->off += len;
		return 0;
	}
}
BENZINA_PUBLIC int          benzinaBufWriteFromFd(BENZINA_BUF*  bbuf,
                                                  int           fd,
                                                  size_t        len){
	ssize_t bytesChunk;
	size_t  bytesLeft = len, bytesRead = 0;
	
	if(benzinaBufEnsure(bbuf, bytesLeft) < 0){
		return -1;
	}
	do{
		bytesChunk = read(fd, bbuf->buf+bbuf->off, bytesLeft);
		if(bytesChunk  < 0){
			return -1;
		}
		bytesRead += bytesChunk;
		bbuf->off += bytesChunk;
		bbuf->len  = bbuf->off > bbuf->len ? bbuf->off : bbuf->len;
		bytesLeft -= bytesChunk;
	}while(bytesChunk != 0 && bytesLeft > 0);
	
	return 0;
}


/* Protobuf */
/* Protobuf Read */
BENZINA_PUBLIC int          benzinaBufReadDelim  (BENZINA_BUF*  bbuf,
                                                  const char**  data,
                                                  size_t*       len){
	uint64_t v;
	int      ret;
	ret = benzinaBufReadvu64(bbuf, &v);
	*len  = v;
	*data = bbuf->buf;
	bbuf->buf += v;
	return ret;
}
BENZINA_PUBLIC int          benzinaBufReadStr    (BENZINA_BUF*  bbuf,
                                                  char**        str,
                                                  size_t*       len){
	int   ret;
	char* nulterm;
	
	ret     = benzinaBufReadDelim(bbuf, (const char**)str, len);
	if(ret){
		return ret;
	}
	
	nulterm = malloc(*len);
	if(nulterm){
		memcpy(nulterm, *str, *len);
		*str = nulterm;
		return 0;
	}else{
		*str = NULL;
		return 1;
	}
}
BENZINA_PUBLIC int          benzinaBufReadMsg    (BENZINA_BUF*  bbuf,
                                                  BENZINA_BUF*  msg){
	int ret;
	
	ret = benzinaBufEnsure(msg, bbuf->len);
	if(ret){
		return ret;
	}
	
	memcpy(msg->buf+msg->len, bbuf->buf, bbuf->len);
	
	return 0;
}

BENZINA_PUBLIC int          benzinaBufReadvu64   (BENZINA_BUF*  bbuf, uint64_t* v){
	char c;
	int  i=0, ret;
	
	*v = 0;
	do{
		ret = benzinaBufRead(bbuf, &c, 1);
		if(ret){
			return ret;
		}
		*v |= (uint64_t)(c&0x7F) << 7*i++;
	}while(i<10 && (c&0x80));
	
	return 0;
}
BENZINA_PUBLIC int          benzinaBufReadvs64   (BENZINA_BUF*  bbuf, int64_t*  v){
	uint64_t w;
	int ret = benzinaBufReadvu64(bbuf, &w);
	*v = (w >> 1)^-(w&1);
	return ret;
}
BENZINA_PUBLIC int          benzinaBufReadvi64   (BENZINA_BUF*  bbuf, int64_t*  v){
	return benzinaBufReadvu64(bbuf, (uint64_t*)v);
}
BENZINA_PUBLIC int          benzinaBufReadvu32   (BENZINA_BUF*  bbuf, uint32_t* v){
	uint64_t w;
	int ret = benzinaBufReadvu64(bbuf, &w);
	*v = w;
	return ret;
}
BENZINA_PUBLIC int          benzinaBufReadvs32   (BENZINA_BUF*  bbuf, int32_t*  v){
	uint32_t w;
	int ret = benzinaBufReadvu32(bbuf, &w);
	*v = (w >> 1)^-(w&1);
	return ret;
}
BENZINA_PUBLIC int          benzinaBufReadvi32   (BENZINA_BUF*  bbuf, int32_t*  v){
	return benzinaBufReadvu32(bbuf, (uint32_t*)v);
}

BENZINA_PUBLIC int          benzinaBufReadfu64   (BENZINA_BUF*  bbuf, uint64_t* u){
	return benzinaBufRead(bbuf, (char*)u, sizeof(*u));
}
BENZINA_PUBLIC int          benzinaBufReadfs64   (BENZINA_BUF*  bbuf, int64_t*  s){
	return benzinaBufRead(bbuf, (char*)s, sizeof(*s));
}
BENZINA_PUBLIC int          benzinaBufReadff64   (BENZINA_BUF*  bbuf, double*   f){
	return benzinaBufRead(bbuf, (char*)f, sizeof(*f));
}
BENZINA_PUBLIC int          benzinaBufReadfu32   (BENZINA_BUF*  bbuf, uint32_t* u){
	return benzinaBufRead(bbuf, (char*)u, sizeof(*u));
}
BENZINA_PUBLIC int          benzinaBufReadfs32   (BENZINA_BUF*  bbuf, int32_t*  s){
	return benzinaBufRead(bbuf, (char*)s, sizeof(*s));
}
BENZINA_PUBLIC int          benzinaBufReadff32   (BENZINA_BUF*  bbuf, float*    f){
	return benzinaBufRead(bbuf, (char*)f, sizeof(*f));
}

BENZINA_PUBLIC int          benzinaBufReadEnum   (BENZINA_BUF*  bbuf, int32_t*  e){
	return benzinaBufReadvi32(bbuf, e);
}
BENZINA_PUBLIC int          benzinaBufReadBool   (BENZINA_BUF*  bbuf, int*      b){
	uint32_t v;
	int ret = benzinaBufReadvu32(bbuf, &v);
	*b = !!v;
	return ret;
}

BENZINA_PUBLIC int          benzinaBufReadTagW   (BENZINA_BUF*  bbuf,
                                                  uint32_t*     tag,
                                                  uint32_t*     wire){
	uint32_t v;
	int ret = benzinaBufReadvu32(bbuf, &v);
	*tag  = v >> 3;
	*wire = v &  0x7;
	return ret;
}

BENZINA_PUBLIC int          benzinaBufReadSkip   (BENZINA_BUF*  bbuf,
                                                  uint32_t      wire){
	uint32_t u32;
	uint64_t u64;
	
	switch(wire){
		case 0: return benzinaBufReadvu64(bbuf, &u64);
		case 1: return benzinaBufReadfu64(bbuf, &u64);
		case 5: return benzinaBufReadfu32(bbuf, &u32);
		case 2:
			if(benzinaBufReadvu64(bbuf, &u64) < 0){
				return -1;
			}
			if(bbuf->off+u64 > bbuf->len){
				return -1;
			}else{
				bbuf->off += u64;
				return 0;
			}
		break;
		default:
			return -1;
	}
}
